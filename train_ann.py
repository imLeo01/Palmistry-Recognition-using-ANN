# train_ann_seg_mlp_oriented_keras.py
# Chuyển từ PyTorch -> Keras Sequential
# - ANN (MLP-Mixer) cho segmentation đường chỉ tay
# - Giữ canonicalize (upright) bằng OpenCV
# - Focal-Tversky loss
# - Gán nhãn HEART/HEAD/LIFE/FATE bằng luật khi eval, lưu overlay + JSON
# Yêu cầu: pip install tensorflow opencv-python albumentations h5py numpy

from pathlib import Path
import os, json, math, numpy as np, cv2, h5py, random
from typing import Tuple, Dict, List

import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import albumentations as A

# ================== CONFIG ==================
ROOT = Path(r"C:\Users\Lazycat\Documents\AI\Number recognition\palmistry")
DATA_DIR   = ROOT / "data"
CKPT_DIR   = ROOT / "checkpoints"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES    = ROOT / "samples"; SAMPLES.mkdir(parents=True, exist_ok=True)

CKPT_H5    = CKPT_DIR / "palm_ann_mlp_oriented_best.h5"
IMAGE_SIZE = 256
PATCH_SIZE = 16
EPOCHS     = 40
BATCH_SIZE = 8
BASE_LR    = 1e-3
PATIENCE   = 8
SEED       = 1337
DEVICE     = "cpu"  # "cpu" or "gpu"
# ============================================

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
set_seed()

# ---------------- Orientation utils (giữ nguyên logic) ----------------
def skin_mask_ycrcb(img_rgb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
    mask = cv2.medianBlur(mask, 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
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
    angle = np.degrees(np.arctan2(main[1], main[0]))
    return float(angle), (float(mean[0]), float(mean[1]))

def rotate_image_and_mask(img_rgb: np.ndarray, mask: np.ndarray, angle_deg: float, center) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    h,w = img_rgb.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    img_r = cv2.warpAffine(img_rgb, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    msk_r = cv2.warpAffine(mask,    M, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    Minv = cv2.invertAffineTransform(M)
    return img_r, msk_r, M, Minv

def cross_width(mask: np.ndarray, center, direction: int, band_frac: float = 0.08) -> float:
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
    u = np.array([v[1], -v[0]])
    width = (band_pts@u).max() - (band_pts@u).min()
    return float(abs(width))

def canonicalize_upright_pair(img_rgb: np.ndarray, mask_u8: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
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

# ---------------- Dataset -> tf.data ----------------
class SegDataLoader:
    IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".webp"}
    def __init__(self, root: Path, subset: str, size=IMAGE_SIZE, aug=False, canonicalize=True):
        self.root = Path(root); self.subset=subset; self.size=size; self.aug=aug; self.canon=canonicalize
        self.img_dir = self.root/subset/"images"; self.msk_dir = self.root/subset/"masks"
        imgs = [p for p in self.img_dir.iterdir() if p.suffix.lower() in self.IMG_EXTS]
        msk_map = {p.stem.lower(): p for p in self.msk_dir.iterdir() if p.suffix.lower() in self.IMG_EXTS}
        self.files = [(im, msk_map.get(im.stem.lower())) for im in imgs if im.stem.lower() in msk_map]
        if not self.files:
            raise RuntimeError(f"{subset}: không có cặp ảnh-mask hợp lệ.")
        # Albumentations pipelines
        self.tf_tr = A.Compose([
            A.Resize(size,size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.35),
            A.Sharpen(alpha=(0.05,0.15), lightness=(0.9,1.1), p=0.25),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.Affine(scale=(0.9,1.1), rotate=(-12,12), translate_percent=(0,0.04), mode=cv2.BORDER_REFLECT_101, p=0.35),
            A.GaussNoise(var_limit=(5,25), p=0.10),
        ])
        self.tf_val = A.Compose([
            A.Resize(size,size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=1.0),
        ])

    def _load_pair(self, ip: Path, mp: Path):
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if self.canon:
            img, msk, _, _ = canonicalize_upright_pair(img, msk)
        return img, msk

    def generator(self):
        for ip, mp in self.files:
            img, msk = self._load_pair(ip, mp)
            if self.aug and self.subset=="train":
                out = self.tf_tr(image=img, mask=msk)
                img2, msk2 = out["image"], out["mask"]
            else:
                out = self.tf_val(image=img, mask=msk)
                img2, msk2 = out["image"], out["mask"]
            # Normalize to 0..1 float32
            img2 = img2.astype(np.float32) / 255.0
            thr = 0.5 if (msk2.max() <= 1) else 127.5
            msk_bin = (msk2 > thr).astype(np.float32)
            msk_bin = np.expand_dims(msk_bin, -1)  # H,W,1
            yield img2, msk_bin, ip.stem

    def tf_dataset(self):
        gen = lambda: self.generator()
        ds = tf.data.Dataset.from_generator(gen,
            output_signature=(
                tf.TensorSpec(shape=(self.size,self.size,3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.size,self.size,1), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string)
            ))
        if self.subset=="train":
            ds = ds.shuffle(128, seed=SEED)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

# ---------------- Loss & metrics ----------------
def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    def loss_fn(y_true, y_pred):
        # y_pred assumed prob in [0,1]
        p = tf.reshape(y_pred, [-1, tf.shape(y_pred)[1]*tf.shape(y_pred)[2]])
        y = tf.reshape(y_true, [-1, tf.shape(y_true)[1]*tf.shape(y_true)[2]])
        tp = tf.reduce_sum(p * y, axis=1)
        fp = tf.reduce_sum(p * (1 - y), axis=1)
        fn = tf.reduce_sum((1 - p) * y, axis=1)
        tversky = (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)
        ft = tf.pow((1 - tversky), gamma)
        return tf.reduce_mean(ft)
    return loss_fn

def iou_metric(threshold=0.5):
    def iou(y_true, y_pred):
        p = tf.cast(y_pred >= threshold, tf.float32)
        inter = tf.reduce_sum(p * y_true, axis=[1,2,3])
        union = tf.reduce_sum(p + y_true - p * y_true, axis=[1,2,3]) + 1e-6
        return tf.reduce_mean(inter/union)
    return iou

def dice_metric(threshold=0.5):
    def dice(y_true, y_pred):
        p = tf.cast(y_pred >= threshold, tf.float32)
        inter = 2.0 * tf.reduce_sum(p * y_true, axis=[1,2,3])
        denom = tf.reduce_sum(p, axis=[1,2,3]) + tf.reduce_sum(y_true, axis=[1,2,3]) + 1e-6
        return tf.reduce_mean(inter/denom)
    return dice

# ---------------- Keras: MLP-Mixer components ----------------
class PatchEmbed(Layer):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=PATCH_SIZE, embed_dim=256, **kwargs):
        super().__init__(**kwargs)
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.proj = layers.Dense(embed_dim)

    def call(self, x):  # x: (B,H,W,3)
        batch = tf.shape(x)[0]
        p = self.patch_size
        # Extract patches: use tf.image.extract_patches
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1,p,p,1],
            strides=[1,p,p,1],
            rates=[1,1,1,1],
            padding='VALID'
        )  # (B, Gh, Gw, p*p*3)
        # flatten patches into (B, N, patch_dim)
        patch_dim = patches.shape[-1]
        N = tf.shape(patches)[1]*tf.shape(patches)[2]
        patches_reshaped = tf.reshape(patches, (batch, N, patch_dim))
        return self.proj(patches_reshaped)  # (B,N,embed)

    def get_config(self):
        cfg = super().get_config(); cfg.update({"patch_size":self.patch_size,"embed_dim":self.embed_dim}); return cfg

class MLPBlock(Layer):
    def __init__(self, hidden_dim, out_dim, drop_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(hidden_dim, activation='gelu')
        self.fc2 = layers.Dense(out_dim)
        self.drop = layers.Dropout(drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x

class MixerBlock(Layer):
    def __init__(self, num_tokens, channels, token_mlp_dim, channel_mlp_dim, drop_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.channels = channels
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.token_mlp = MLPBlock(token_mlp_dim, num_tokens, drop_rate)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.channel_mlp = MLPBlock(channel_mlp_dim, channels, drop_rate)

    def call(self, x, training=False):
        # x: (B, N, C)
        y = self.ln1(x)
        y = tf.transpose(y, perm=[0,2,1])  # (B, C, N)
        y = self.token_mlp(y, training=training)
        y = tf.transpose(y, perm=[0,2,1])
        x = x + y
        y2 = self.ln2(x)
        y2 = self.channel_mlp(y2, training=training)
        return x + y2

    def get_config(self):
        cfg = super().get_config(); cfg.update({"num_tokens":self.num_tokens,"channels":self.channels}); return cfg

class MLPMixerSegModel(Layer):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_ch=3, embed=256, depth=9, t_mlp=128, c_mlp=512, **kwargs):
        super().__init__(**kwargs)
        assert img_size % patch_size == 0
        self.img_size = img_size; self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_tokens = self.grid * self.grid
        self.embed = embed
        # Patch embed
        self.patch_embed = PatchEmbed(img_size, patch_size, embed)
        # Mixer blocks (as a Sequential inside)
        self.blocks = [MixerBlock(self.num_tokens, embed, t_mlp, c_mlp) for _ in range(depth)]
        self.ln = layers.LayerNormalization(epsilon=1e-6)
        self.head = layers.Dense(1)  # token-level prediction

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        tok = self.patch_embed(x)  # (B,N,embed)
        for blk in self.blocks:
            tok = blk(tok, training=training)
        tok = self.ln(tok)  # (B,N,embed)
        logits_tok = self.head(tok)  # (B,N,1)
        logits_tok = tf.squeeze(logits_tok, axis=-1)  # (B,N)
        # reshape to grid and upsample to image
        grid = tf.reshape(logits_tok, (B, self.grid, self.grid, 1))  # (B, Gh, Gw, 1)
        up = tf.image.resize(grid, size=(self.img_size, self.img_size), method='bilinear', antialias=True)
        return up  # logits map (B,H,W,1)

    def get_config(self):
        cfg = super().get_config(); cfg.update({"img_size":self.img_size,"patch_size":self.patch_size,"embed":self.embed}); return cfg

# ---------------- Rules: label 4 lines on canonical images (giữ nguyên) ----------------
def classify_major_lines(mask_u8: np.ndarray) -> Dict[str, np.ndarray]:
    h,w = mask_u8.shape[:2]
    m = (mask_u8>127).astype(np.uint8)
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
        ang = np.degrees(np.arctan2(main[1], main[0]))
        lines.append({'bbox':(x,y,ww,hh),'center':(cx,cy),'angle':ang,'pts':pts,'len':L})
    top_band  = 0.35*h
    mid_band  = 0.60*h
    left_edge = 0.22*w
    left_thumb= 0.40*w
    mid_x     = 0.50*w
    out = {}
    cand = [L for L in lines if L['center'][1] < top_band and abs(L['angle']) < 30]
    if cand: out['heart'] = max(cand, key=lambda L: L['len'])['pts']
    cand = [L for L in lines if top_band <= L['center'][1] <= mid_band and abs(L['angle']) < 30]
    if cand: out['head']  = max(cand, key=lambda L: L['len'])['pts']
    cand = [L for L in lines if L['center'][0] < left_thumb and L['bbox'][0] < left_edge]
    if cand: out['life']  = max(cand, key=lambda L: L['len'])['pts']
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

# ---------------- Training loop (Keras .fit) ----------------
def train():
    # Prepare datasets
    tr = SegDataLoader(DATA_DIR, "train", IMAGE_SIZE, aug=True, canonicalize=True)
    va = SegDataLoader(DATA_DIR, "val",   IMAGE_SIZE, aug=False, canonicalize=True)
    te = SegDataLoader(DATA_DIR, "test",  IMAGE_SIZE, aug=False, canonicalize=True)

    ds_tr = tr.tf_dataset()
    ds_va = va.tf_dataset()
    ds_te = te.tf_dataset()

    # Build model as Sequential with custom layer wrapper
    mixer_layer = MLPMixerSegModel(img_size=IMAGE_SIZE, patch_size=PATCH_SIZE, embed=256, depth=9, t_mlp=128, c_mlp=512)
    model = Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        mixer_layer,
        layers.Activation('sigmoid')  # output probability map (B,H,W,1)
    ])
    model.summary()

    # Compile
    loss_fn = focal_tversky_loss(alpha=0.7, beta=0.3, gamma=0.75)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR),
                  loss=loss_fn,
                  metrics=[iou_metric(0.5), dice_metric(0.5)])

    # Callbacks
    ckpt = ModelCheckpoint(str(CKPT_H5), monitor='val_iou', mode='max', save_best_only=True, save_weights_only=False)
    early = EarlyStopping(monitor='val_iou', mode='max', patience=PATIENCE, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, mode='min')

    # Fit
    model.fit(ds_tr.map(lambda x,y,z: (x,y)),
              validation_data=ds_va.map(lambda x,y,z: (x,y)),
              epochs=EPOCHS,
              callbacks=[ckpt, early, reduce_lr])

    # Evaluate on test
    res = model.evaluate(ds_te.map(lambda x,y,z: (x,y)))
    print("TEST results (loss, iou, dice):", res)

    # Save overlays & JSON labels similar logic
    (SAMPLES / "overlays").mkdir(parents=True, exist_ok=True)
    labels_json = {}
    for batch in ds_te:
        imgs, masks, metas = batch
        probs = model.predict(imgs)  # [B,H,W,1]
        probs = (probs[...,0] >= 0.5).astype(np.uint8) * 255
        imgs_np = (imgs.numpy() * 255).astype(np.uint8)
        for b in range(probs.shape[0]):
            mid = metas[b].numpy().decode('utf-8')
            m = probs[b]
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
            L = classify_major_lines(m)
            labels_json[mid] = {k: v.astype(int).tolist() for k,v in L.items()}
            # overlay uses image from tensor (RGB)
            img = imgs_np[b]
            out = overlay_and_draw(img, m, L, alpha=0.35)
            cv2.imwrite(str(SAMPLES/"overlays"/f"{mid}_overlay.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    with open(SAMPLES / "test_line_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_json, f, ensure_ascii=False, indent=2)
    print(f"Saved overlays -> {SAMPLES/'overlays'}")
    print(f"Saved line labels JSON -> {SAMPLES/'test_line_labels.json'}")
    print(f"Saved model -> {CKPT_H5}")

# ================== ENTRY ==================
if __name__=="__main__":
    assert IMAGE_SIZE % PATCH_SIZE == 0, "IMAGE_SIZE phải chia hết cho PATCH_SIZE"
    # Set device (cpu/gpu)
    if DEVICE.lower() == "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    train()
