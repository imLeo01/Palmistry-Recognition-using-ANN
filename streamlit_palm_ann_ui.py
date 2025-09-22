# streamlit_palmistry_app.py
# UI Palmistry: dùng ANN (MLP-Mixer, NO CNN) đúng với file train_ann_seg_mlp_palmistry.py
# - Load .pt/.h5
# - Segment mask -> phân loại 4 đường (heart, head, life, fate) -> vẽ & hiển thị

from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
import albumentations as A

# =========================
# CONFIG (đổi nếu cần)
# =========================
ROOT = Path(r"C:\Users\Lazycat\Documents\AI\Number recognition\palmistry")
CKPT_PT = ROOT / "checkpoints" / "palm_ann_mlp_best.pt"
CKPT_H5 = ROOT / "checkpoints" / "palm_ann_mlp_best.h5"
DEFAULT_IMAGE_SIZE = 320
DEFAULT_PATCH_SIZE = 16
DEVICE = "cpu"   # ép CPU
# =========================


# ---------- ANN model (MLP-Mixer) – giống hệt file train ----------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class MixerBlock(nn.Module):
    def __init__(self, num_tokens, channels, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(channels)
        self.token_mlp = Mlp(num_tokens, token_mlp_dim, num_tokens)
        self.ln2 = nn.LayerNorm(channels)
        self.channel_mlp = Mlp(channels, channel_mlp_dim, channels)
    def forward(self, x):  # x: (B,N,C)
        y = self.ln1(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + y
        y = self.ln2(x)
        return x + self.channel_mlp(y)

class MLPMixerSeg(nn.Module):
    def __init__(self, img_size=320, patch_size=16, in_ch=3, embed=256, depth=8, t_mlp=128, c_mlp=512):
        super().__init__()
        assert img_size % patch_size == 0
        self.ps = patch_size
        self.grid = img_size // patch_size
        num_tokens = self.grid * self.grid
        patch_dim = in_ch * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed)
        self.blocks = nn.Sequential(*[
            MixerBlock(num_tokens, embed, t_mlp, c_mlp) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, 1)

    def _to_patches(self, x):  # (B,3,H,W)->(B,N,3*p*p)
        p = torch.nn.functional.unfold(x, kernel_size=self.ps, stride=self.ps)  # (B,C*p*p,N)
        return p.transpose(1, 2)

    def forward(self, x):
        B, _, H, W = x.shape
        tok = self.patch_embed(self._to_patches(x))
        tok = self.blocks(tok)
        tok = self.ln(tok)
        logits_tok = self.head(tok).squeeze(-1)  # (B,N)
        grid = logits_tok.view(B, 1, self.grid, self.grid)
        return torch.nn.functional.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)


# ---------- utils: load weights ----------
def load_pt(ckpt_path: Path, model: nn.Module):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    state = ck.get("model", ck)
    model.load_state_dict(state)

def load_h5(h5_path: Path, model: nn.Module):
    import h5py
    with h5py.File(str(h5_path), "r") as f:
        sd = {k: torch.tensor(f[k][...]) for k in f.keys()}
    model.load_state_dict(sd)

# ---------- preprocessing ----------
def build_transforms(size):
    return A.Compose([A.Resize(size, size, interpolation=cv2.INTER_LINEAR), A.Normalize()])

@torch.no_grad()
def infer_mask(model, pil_img: Image.Image, size=320, thr=0.5):
    tf = build_transforms(size)
    img = np.array(pil_img.convert("RGB"))
    x = tf(image=img)["image"].transpose(2, 0, 1)  # HWC->CHW
    x = torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob >= thr).astype(np.uint8) * 255
    return mask  # size×size

def normalize_mask(mask, target_hw):
    m = np.asarray(mask)
    if m.ndim == 3:
        if m.shape[2] == 3: m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        else: m = np.squeeze(m, -1)
    if m.dtype == bool:
        m = (m.astype(np.uint8) * 255)
    elif np.issubdtype(m.dtype, np.floating):
        m = (np.clip(m, 0, 1) * 255.0).round().astype(np.uint8)
    elif m.dtype != np.uint8:
        m = m.astype(np.uint8)
    H, W = target_hw
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m

def apply_post(mask, close_ks=3, open_ks=0, thin=False):
    m = mask.copy()
    if close_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    if open_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    if thin:
        try:
            import skimage.morphology as skm
            m = (skm.skeletonize((m > 0).astype(bool)).astype(np.uint8) * 255)
        except Exception:
            pass
    return m


# ---------- RULES: phân loại heart/head/life/fate (giống file train) ----------
def classify_major_lines(mask_u8: np.ndarray):
    h, w = mask_u8.shape[:2]
    m = (mask_u8 > 127).astype(np.uint8)

    # Skeleton
    try:
        import skimage.morphology as skm
        sk = (skm.skeletonize(m > 0).astype(np.uint8)) * 255
    except Exception:
        sk = m

    cnts, _ = cv2.findContours(sk, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    lines = []
    for c in cnts:
        length = cv2.arcLength(c, False)
        if length < 0.05 * min(h, w):
            continue
        pts = c[:, 0, :]
        x, y, wc, hc = cv2.boundingRect(pts)
        cx, cy = x + wc / 2, y + hc / 2

        # hướng chính (PCA)
        mu = cv2.moments(pts)
        if mu["m00"] == 0: 
            continue
        xs = pts[:, 0] - cx
        ys = pts[:, 1] - cy
        cov = np.cov(np.stack([xs, ys], 0))
        vals, vecs = np.linalg.eig(cov)
        main = vecs[:, np.argmax(vals)]
        angle = np.degrees(np.arctan2(main[1], main[0]))  # 0° ngang, ±90° dọc

        lines.append({'bbox': (x, y, wc, hc), 'center': (cx, cy), 'angle': angle, 'pts': pts, 'len': length})

    top_band = 0.35 * h
    mid_band = 0.55 * h
    left_thumb = 0.45 * w
    mid_x = 0.5 * w

    out = {}

    # HEART: trên, ngang
    cand = [L for L in lines if L['center'][1] < top_band and abs(L['angle']) < 30]
    if cand: out['heart'] = max(cand, key=lambda L: L['len'])['pts']

    # HEAD: giữa, ngang
    cand = [L for L in lines if top_band <= L['center'][1] <= mid_band and abs(L['angle']) < 30]
    if cand: out['head']  = max(cand, key=lambda L: L['len'])['pts']

    # LIFE: trái (vùng ngón cái), bám mép trái
    cand = [L for L in lines if L['center'][0] < left_thumb and L['bbox'][0] < 0.25 * w]
    if cand: out['life']  = max(cand, key=lambda L: L['len'])['pts']

    # FATE: giữa, dọc
    cand = [L for L in lines if abs(L['center'][0] - mid_x) < 0.15 * w and abs(abs(L['angle']) - 90) < 25]
    if cand: out['fate']  = max(cand, key=lambda L: L['len'])['pts']

    return out


# ---------- overlay ----------
def overlay_green(img_rgb, mask_gray, alpha=0.35):
    mask_col = np.zeros_like(img_rgb, dtype=np.uint8)
    mask_col[..., 1] = mask_gray
    return cv2.addWeighted(img_rgb, 1.0, mask_col, alpha, 0)

def draw_labeled_lines(img_rgb, lines_dict, color_map=None):
    out = img_rgb.copy()
    if color_map is None:
        color_map = {
            'heart': (255, 80, 80),  # BGR đỏ nhạt
            'head' : (80, 180, 255), # cam
            'life' : (80, 255, 120), # xanh lá
            'fate' : (200, 80, 255), # hồng
        }
    for name, pts in lines_dict.items():
        pts = pts.astype(np.int32)
        cv2.polylines(out, [pts], isClosed=False, color=color_map.get(name,(0,255,0)), thickness=2)
        # vẽ nhãn ở điểm đầu polyline
        p0 = tuple(pts[0])
        cv2.putText(out, name.upper(), p0, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map.get(name,(0,255,0)), 2, cv2.LINE_AA)
    return out


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Palmistry ANN (MLP) — Detection", layout="wide")
st.markdown(
    """
    <h2 style="margin:0;padding:12px 16px;border-radius:12px;
               background:linear-gradient(90deg,#7ac4d6,#6db6cc,#6bb3c9);
               color:#08323d;">Palmistry</h2>
    """,
    unsafe_allow_html=True
)

left, right = st.columns([1,1.05])

with left:
    st.subheader("Tải ảnh & Chọn checkpoint")
    file = st.file_uploader("Ảnh bàn tay", type=["png","jpg","jpeg","bmp","webp"])
    ckcol1, ckcol2 = st.columns(2)
    ckpt_pt = ckcol1.text_input("Checkpoint (.pt)", str(CKPT_PT))
    ckpt_h5 = ckcol2.text_input("Checkpoint (.h5)", str(CKPT_H5))

    colA, colB, colC = st.columns(3)
    image_size = colA.number_input("IMAGE_SIZE", 128, 1024, DEFAULT_IMAGE_SIZE, step=16)
    patch_size = colB.number_input("PATCH_SIZE", 4, 64, DEFAULT_PATCH_SIZE, step=2)
    thr        = colC.slider("Threshold", 0.1, 0.9, 0.5, 0.01)

    close_ks = st.slider("Morph CLOSE (kernel)", 0, 15, 3, 1)
    open_ks  = st.slider("Morph OPEN (kernel)", 0, 15, 0, 1)
    thin     = st.checkbox("Skeletonize (mảnh hóa)", value=False)
    alpha    = st.slider("Overlay alpha", 0.1, 0.9, 0.35, 0.05)

    run = st.button("🚀 Detect (ANN)")

with right:
    st.subheader("Các đường chỉ tay cho biết bạn:")
    st.markdown(
        """
        <ul style="font-size:16px; line-height:1.5;">
          <li>Heart line (tình cảm): gần gốc ngón tay, chạy ngang.</li>
          <li>Head line (trí tuệ): ở giữa lòng bàn tay, chạy ngang.</li>
          <li>Life line (sức khỏe): cong ôm gốc ngón cái.</li>
          <li>Fate line (sự nghiệp): dọc gần giữa lòng bàn tay.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

# cache model
@st.cache_resource(show_spinner=True)
def get_model(img_size:int, patch:int, pt_path:str, h5_path:str):
    assert img_size % patch == 0, "IMAGE_SIZE phải chia hết cho PATCH_SIZE"
    model = MLPMixerSeg(img_size=img_size, patch_size=patch).to(DEVICE).eval()
    p_pt, p_h5 = Path(pt_path), Path(h5_path)
    if p_pt.exists():
        load_pt(p_pt, model); return model, ".pt"
    if p_h5.exists():
        load_h5(p_h5, model); return model, ".h5"
    raise FileNotFoundError("Không tìm thấy checkpoint .pt hoặc .h5")

if run and file:
    try:
        model, fmt = get_model(int(image_size), int(patch_size), ckpt_pt, ckpt_h5)
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        st.stop()

    pil = Image.open(file).convert("RGB")
    img_rgb = np.array(pil)  # (H,W,3) RGB
    H, W = img_rgb.shape[:2]

    with st.spinner("Đang phân đoạn bằng ANN…"):
        mask_small = infer_mask(model, pil, size=int(image_size), thr=float(thr))
        mask_small = apply_post(mask_small, close_ks=int(close_ks), open_ks=int(open_ks), thin=bool(thin))
        mask_u8 = normalize_mask(mask_small, (H, W))

    # phân loại 4 đường
    with st.spinner("Đang gán nhãn Heart/Head/Life/Fate…"):
        lines = classify_major_lines(mask_u8)

    # hiển thị
    c1, c2, c3 = st.columns([1,1,1])
    c1.image(img_rgb, caption="Ảnh gốc", use_container_width=True)
    c2.image(mask_u8, caption="Mask (ANN)", use_container_width=True, clamp=True)
    over = overlay_green(img_rgb, mask_u8, alpha=float(alpha))
    over_lbl = draw_labeled_lines(over, lines)
    c3.image(over_lbl, caption=f"Overlay + nhãn ({fmt})", use_container_width=True)
