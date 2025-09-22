# train_ann_seg_mlp_oriented.py
# ANN (MLP-Mixer, NO CNN) cho segmentation đường chỉ tay
# + Chuẩn hoá phương hướng bàn tay (upright)
# + Focal-Tversky loss
# + Gán nhãn HEART/HEAD/LIFE/FATE bằng luật khi eval, lưu overlay + JSON

from pathlib import Path
import os, json, math, numpy as np, cv2, h5py
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================== CONFIG ==================
ROOT = Path(r"C:\Users\Lazycat\Documents\AI\Number recognition\palmistry")
DATA_DIR   = ROOT / "data"    
CKPT_DIR   = ROOT / "checkpoints"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES    = ROOT / "samples"; SAMPLES.mkdir(parents=True, exist_ok=True)

CKPT_PT    = CKPT_DIR / "palm_ann_mlp_oriented_best.pt"
CKPT_H5    = CKPT_DIR / "palm_ann_mlp_oriented_best.h5"

IMAGE_SIZE = 256
PATCH_SIZE = 16
EPOCHS     = 40
BATCH_SIZE = 8
BASE_LR    = 1e-3
WEIGHT_DEC = 1e-4
PATIENCE   = 8
SEED       = 1337
DEVICE     = "cpu"          # ép CPU
# ============================================

def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ---------------- Orientation utils ----------------
def skin_mask_ycrcb(img_rgb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    # Ngưỡng phổ biến cho da (rộng để bao phủ nhiều tông)
    mask = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
    mask = cv2.medianBlur(mask, 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    # lấy blob lớn nhất
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return np.zeros_like(mask)
    c = max(cnts, key=cv2.contourArea)
    big = np.zeros_like(mask); cv2.drawContours(big, [c], -1, 255, thickness=cv2.FILLED)
    return big

def estimate_orientation(mask: np.ndarray) -> Tuple[float, Tuple[float,float]]:
    ys, xs = np.where(mask>0)
    if len(xs) < 10:
        h,w = mask.shape[:2]
        return 0.0, (w/2, h/2)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    cov = np.cov((pts-mean).T)
    vals, vecs = np.linalg.eigh(cov)
    main = vecs[:, np.argmax(vals)]
    angle = np.degrees(np.arctan2(main[1], main[0]))  # trục CHÍNH so với Ox
    return float(angle), (float(mean[0]), float(mean[1]))

def rotate_image_and_mask(img_rgb: np.ndarray, mask: np.ndarray, angle_deg: float, center) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    h,w = img_rgb.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    img_r = cv2.warpAffine(img_rgb, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    msk_r = cv2.warpAffine(mask,    M, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    Minv = cv2.invertAffineTransform(M)
    return img_r, msk_r, M, Minv

def cross_width(mask: np.ndarray, center, direction: int, band_frac: float = 0.08) -> float:
    """Ước lượng bề ngang gần đầu mút theo trục chính để phân biệt ngón tay vs cổ tay."""
    ys, xs = np.where(mask>0)
    if len(xs)<10: return 0.0
    ang,_ = estimate_orientation(mask)
    v = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))], np.float32) * direction
    pts = np.stack([xs, ys], 1).astype(np.float32)
    proj = (pts - np.array(center)) @ v
    t_end = np.percentile(proj, 98)
    sel = proj > (t_end - band_frac*max(mask.shape))
    band_pts = pts[sel]
    if len(band_pts)<5: return 0.0
    u = np.array([v[1], -v[0]])  # vuông góc
    width = (band_pts@u).max() - (band_pts@u).min()
    return float(abs(width))

def canonicalize_upright_pair(img_rgb: np.ndarray, mask_u8: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Xoay ảnh & mask GT về tư thế chuẩn: ngón tay hướng lên (fingers UP)."""
    skin = skin_mask_ycrcb(img_rgb)
    ang, ctr = estimate_orientation(skin if skin.any() else mask_u8)
    rot1 = (90 - ang)
    img1, msk1, M1, Minv1 = rotate_image_and_mask(img_rgb, mask_u8, rot1, ctr)
    skin1 = skin_mask_ycrcb(img1)
    ang1, ctr1 = estimate_orientation(skin1 if skin1.any() else msk1)
    w_fingers = cross_width(skin1 if skin1.any() else msk1, ctr1, direction=-1)
    w_wrist   = cross_width(skin1 if skin1.any() else msk1, ctr1, direction=+1)
    if w_fingers < w_wrist:
        img2, msk2, M2, Minv2 = rotate_image_and_mask(img1, msk1, 180, ctr1)
        M = np.vstack([M2,[0,0,1]]) @ np.vstack([M1,[0,0,1]]); Minv = np.linalg.inv(M)
        return img2, msk2, M[:2], Minv[:2]
    return img1, msk1, M1, Minv1

# ---------------- Dataset ----------------
class SegDataset(Dataset):
    IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".webp"}
    def __init__(self, root: Path, subset: str, size=384, aug=False, canonicalize=True):
        self.root = Path(root); self.subset=subset; self.size=size; self.aug=aug; self.canon=canonicalize
        self.img_dir = self.root/subset/"images"; self.msk_dir = self.root/subset/"masks"
        imgs = [p for p in self.img_dir.iterdir() if p.suffix.lower() in self.IMG_EXTS]
        msk_map = {p.stem.lower(): p for p in self.msk_dir.iterdir() if p.suffix.lower() in self.IMG_EXTS}
        self.files = [(im, msk_map.get(im.stem.lower())) for im in imgs if im.stem.lower() in msk_map]
        if not self.files: raise RuntimeError(f"{subset}: không có cặp ảnh-mask hợp lệ.")

        # Augment (tăng tương phản rãnh mảnh + biến đổi nhẹ)
        self.tf_tr = A.Compose([
            A.Resize(size,size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.35),
            A.Sharpen(alpha=(0.05,0.15), lightness=(0.9,1.1), p=0.25),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.Affine(scale=(0.9,1.1), rotate=(-12,12), translate_percent=(0,0.04),
                     mode=cv2.BORDER_REFLECT_101, p=0.35),
            A.GaussNoise(var_limit=(5,25), p=0.10),
            A.Normalize(), ToTensorV2(),
        ])
        self.tf_val = A.Compose([
            A.Resize(size,size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=1.0),
            A.Normalize(), ToTensorV2()
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        ip, mp = self.files[i]
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        # Chuẩn hoá định hướng trước khi resize/normalize
        if self.canon:
            img, msk, _, _ = canonicalize_upright_pair(img, msk)
        out = self.tf_tr(image=img, mask=msk) if (self.aug and self.subset=="train") else self.tf_val(image=img, mask=msk)
        x = out["image"]; m = out["mask"]
        thr = 0.5 if (m.max() <= 1) else 127.5
        y = (m > thr).float().unsqueeze(0)  # [1,H,W]
        meta = {"id": ip.stem}
        return x, y, meta

# ---------------- Loss & metrics ----------------
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__(); self.a=alpha; self.b=beta; self.g=gamma; self.s=smooth
    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        tp = (p*y).sum((2,3))
        fp = (p*(1-y)).sum((2,3))
        fn = ((1-p)*y).sum((2,3))
        tversky = (tp + self.s) / (tp + self.a*fp + self.b*fn + self.s)
        ft = (1 - tversky).pow(self.g)
        return ft.mean()

@torch.no_grad()
def eval_metrics(model, dl, device, thr=0.5):
    model.eval(); ious=[]; dices=[]
    for x,y,_ in dl:
        x,y=x.to(device),y.to(device)
        p=(torch.sigmoid(model(x))>thr).float()
        inter=(p*y).sum((2,3))
        union=((p+y)-(p*y)).sum((2,3))
        iou =(inter/(union+1e-6)).mean().item()
        dice=(2*inter/(p.sum((2,3))+y.sum((2,3))+1e-6)).mean().item()
        ious.append(iou); dices.append(dice)
    return float(np.mean(ious)), float(np.mean(dices))

# ---------------- ANN: MLP-Mixer ----------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden, out_features, drop=0.0):
        super().__init__()
        self.fc1=nn.Linear(in_features,hidden); self.act=nn.GELU(); self.fc2=nn.Linear(hidden,out_features); self.drop=nn.Dropout(drop)
    def forward(self,x): return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class MixerBlock(nn.Module):
    def __init__(self, num_tokens, channels, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.ln1=nn.LayerNorm(channels); self.token_mlp=Mlp(num_tokens,token_mlp_dim,num_tokens)
        self.ln2=nn.LayerNorm(channels); self.channel_mlp=Mlp(channels,channel_mlp_dim,channels)
    def forward(self,x):              # x: (B,N,C)
        y=self.ln1(x).transpose(1,2); y=self.token_mlp(y).transpose(1,2); x=x+y
        y=self.ln2(x); return x+self.channel_mlp(y)

class MLPMixerSeg(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_ch=3, embed=256, depth=9, t_mlp=128, c_mlp=512):
        super().__init__()
        assert img_size%patch_size==0
        self.ps=patch_size; self.grid=img_size//patch_size
        patch_dim=in_ch*patch_size*patch_size
        self.patch_embed=nn.Linear(patch_dim, embed)
        self.blocks=nn.Sequential(*[MixerBlock(self.grid*self.grid, embed, t_mlp, c_mlp) for _ in range(depth)])
        self.ln=nn.LayerNorm(embed); self.head=nn.Linear(embed,1)

    def _to_patches(self,x):  # (B,3,H,W)->(B,N,3*p*p)
        p=torch.nn.functional.unfold(x,kernel_size=self.ps,stride=self.ps)  # (B,Cp^2,N)
        return p.transpose(1,2)

    def forward(self,x):
        B,_,H,W=x.shape
        tok=self.patch_embed(self._to_patches(x))
        tok=self.blocks(tok); tok=self.ln(tok)
        logits_tok=self.head(tok).squeeze(-1)          # (B,N)
        grid=logits_tok.view(B,1,self.grid,self.grid)
        return torch.nn.functional.interpolate(grid,size=(H,W),mode="bilinear",align_corners=False)

# ------------- Save state_dict to .h5 -------------
def save_state_dict_to_h5(model: nn.Module, h5_path: Path):
    with h5py.File(str(h5_path), "w") as f:
        for k,v in model.state_dict().items():
            f.create_dataset(k, data=v.detach().cpu().numpy())

# ------------- Rules: label 4 lines on canonical images -------------
def classify_major_lines(mask_u8: np.ndarray) -> Dict[str, np.ndarray]:
    h,w = mask_u8.shape[:2]
    m = (mask_u8>127).astype(np.uint8)
    # skeleton
    try:
        import skimage.morphology as skm
        sk = (skm.skeletonize(m>0).astype(np.uint8))*255
    except Exception:
        sk = m
    cnts,_ = cv2.findContours(sk, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    lines=[]
    for c in cnts:
        L = cv2.arcLength(c, False)
        if L < 0.05*min(h,w): continue
        pts = c[:,0,:]
        x,y,ww,hh = cv2.boundingRect(pts)
        cx,cy = x+ww/2, y+hh/2
        mu = cv2.moments(pts)
        if mu['m00']==0: continue
        X = pts[:,0]-cx; Y = pts[:,1]-cy
        cov = np.cov(np.stack([X,Y],0))
        val, vec = np.linalg.eig(cov)
        main = vec[:, np.argmax(val)]
        ang = np.degrees(np.arctan2(main[1], main[0]))  # 0 ngang, ±90 dọc
        lines.append({'bbox':(x,y,ww,hh),'center':(cx,cy),'angle':ang,'pts':pts,'len':L})

    top_band  = 0.35*h
    mid_band  = 0.60*h
    left_edge = 0.22*w
    left_thumb= 0.40*w
    mid_x     = 0.50*w

    out = {}
    # HEART: trên + ngang
    cand = [L for L in lines if L['center'][1] < top_band and abs(L['angle']) < 30]
    if cand: out['heart'] = max(cand, key=lambda L: L['len'])['pts']
    # HEAD: giữa + ngang
    cand = [L for L in lines if top_band <= L['center'][1] <= mid_band and abs(L['angle']) < 30]
    if cand: out['head']  = max(cand, key=lambda L: L['len'])['pts']
    # LIFE: trái gần ngón cái + sát mép trái
    cand = [L for L in lines if L['center'][0] < left_thumb and L['bbox'][0] < left_edge]
    if cand: out['life']  = max(cand, key=lambda L: L['len'])['pts']
    # FATE: giữa + dọc
    cand = [L for L in lines if abs(L['center'][0]-mid_x) < 0.18*w and abs(abs(L['angle'])-90) < 25]
    if cand: out['fate']  = max(cand, key=lambda L: L['len'])['pts']
    return out

def overlay_and_draw(img_rgb: np.ndarray, mask_u8: np.ndarray, lines: Dict[str,np.ndarray], alpha=0.35) -> np.ndarray:
    color = np.zeros_like(img_rgb, dtype=np.uint8); color[...,1] = mask_u8
    over = cv2.addWeighted(img_rgb, 1.0, color, alpha, 0)
    cmap = {'heart':(80,80,255),'head':(0,165,255),'life':(60,220,60),'fate':(220,60,200)}  # BGR
    out = over.copy()
    for name, pts in lines.items():
        pts = pts.astype(np.int32)
        cv2.polylines(out, [pts], False, cmap.get(name,(0,255,0)), 2, cv2.LINE_AA)
        p0 = tuple(pts[0]); cv2.putText(out, name.upper(), p0, cv2.FONT_HERSHEY_SIMPLEX, 0.7, cmap.get(name,(0,255,0)), 2, cv2.LINE_AA)
    return out

# ------------- Training loop -------------
def train():
    set_seed(SEED)
    tr=SegDataset(DATA_DIR,"train",IMAGE_SIZE,aug=True, canonicalize=True)
    va=SegDataset(DATA_DIR,"val",  IMAGE_SIZE,aug=False,canonicalize=True)
    te=SegDataset(DATA_DIR,"test", IMAGE_SIZE,aug=False,canonicalize=True)

    dl_tr=DataLoader(tr,BATCH_SIZE,shuffle=True,num_workers=0)
    dl_va=DataLoader(va,BATCH_SIZE,shuffle=False,num_workers=0)
    dl_te=DataLoader(te,BATCH_SIZE,shuffle=False,num_workers=0)

    model=MLPMixerSeg(img_size=IMAGE_SIZE, patch_size=PATCH_SIZE).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DEC)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=BASE_LR, steps_per_epoch=max(1,len(dl_tr)), epochs=EPOCHS, pct_start=0.3)
    lossf=FocalTverskyLoss(alpha=0.7,beta=0.3,gamma=0.75)

    best=-1e9; bad=0
    for ep in range(1,EPOCHS+1):
        model.train(); run=0.0; n=0
        for x,y,_ in dl_tr:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits=model(x); loss=lossf(logits,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run+=loss.item(); n+=1

        val_iou,val_dice=eval_metrics(model, dl_va, DEVICE)
        score=(val_iou+val_dice)/2
        print(f"Epoch {ep:03d} | TrainLoss {run/max(1,n):.4f} | Val IoU {val_iou:.4f}  Dice {val_dice:.4f}")

        if score>best:
            best=score; bad=0
            torch.save({'model':model.state_dict(),'size':IMAGE_SIZE,'patch':PATCH_SIZE}, CKPT_PT)
            with h5py.File(str(CKPT_H5), "w") as f:
                for k,v in model.state_dict().items():
                    f.create_dataset(k, data=v.detach().cpu().numpy())
        else:
            bad+=1
        if bad>=PATIENCE:
            print("Early stop."); break

    # ==== Test với best + tạo overlay/JSON labels ====
    ck=torch.load(CKPT_PT,map_location=DEVICE); model.load_state_dict(ck['model']); model.eval()
    ti,td=eval_metrics(model, dl_te, DEVICE)
    print(f"TEST | IoU {ti:.4f}  Dice {td:.4f}\nSaved: {CKPT_PT}  {CKPT_H5}")

    # Tạo mẫu minh hoạ & JSON polyline label
    (SAMPLES / "overlays").mkdir(parents=True, exist_ok=True)
    labels_json = {}
    for x,y,meta in dl_te:
        x = x.to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).cpu().numpy()  # [B,1,H,W]
        for b in range(x.size(0)):
            mid = meta["id"][b]
            m = (prob[b,0] >= 0.5).astype(np.uint8)*255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
            # Vì dữ liệu trong loader đã được canonicalize, có thể gán nhãn trực tiếp
            L = classify_major_lines(m)
            labels_json[mid] = {k: v.astype(int).tolist() for k,v in L.items()}

            # Lưu overlay tham khảo (dùng ảnh từ tensor để đảm bảo khớp kích thước)
            img = (x[b].cpu().numpy().transpose(1,2,0))
            img = ((img - img.min())/(img.max()-img.min()+1e-8)*255).astype(np.uint8)
            out = overlay_and_draw(img, m, L, alpha=0.35)
            cv2.imwrite(str(SAMPLES/"overlays"/f"{mid}_overlay.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    with open(SAMPLES / "test_line_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_json, f, ensure_ascii=False, indent=2)
    print(f"Saved overlays -> {SAMPLES/'overlays'}")
    print(f"Saved line labels JSON -> {SAMPLES/'test_line_labels.json'}")

# ================== ENTRY ==================
if __name__=="__main__":
    assert IMAGE_SIZE % PATCH_SIZE == 0, "IMAGE_SIZE phải chia hết cho PATCH_SIZE"
    train()
