import os, shutil, random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def copy_with_rename(src: Path, dst_dir: Path):
    """Copy file; nếu trùng tên thì thêm _1, _2, ..."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    base, ext = os.path.splitext(src.name)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    k = 1
    while True:
        cand = dst_dir / f"{base}_{k}{ext}"
        if not cand.exists():
            shutil.copy2(src, cand)
            return cand
        k += 1

def gather_unique_by_stem(roots):
    """
    Duyệt nhiều roots, gom file theo stem (không đuôi).
    Nếu trùng stem giữa các roots, lấy file gặp đầu tiên.
    """
    mapping = {}  # stem -> Path
    for root in roots:
        root = Path(root)
        if not root.exists(): 
            print(f"⚠️ Bỏ qua (không tồn tại): {root}")
            continue
        for p in list_images(root):
            st = p.stem.lower()
            if st not in mapping:
                mapping[st] = p
    return mapping  # dict(stem->Path)

def make_pairs(img_roots, mask_roots):
    """
    Trả về list[(img_path, mask_path|None)] ghép theo stem.
    Ưu tiên file đầu tiên theo thứ tự roots.
    """
    img_map = gather_unique_by_stem(img_roots)
    mask_map = gather_unique_by_stem(mask_roots) if mask_roots else {}
    pairs = []
    for stem, img in img_map.items():
        mask = mask_map.get(stem)
        pairs.append((img, mask))
    return pairs

def split_pairs(pairs, train=0.7, val=0.15, test=0.15, seed=1337):
    assert abs(train + val + test - 1.0) < 1e-6
    rng = random.Random(seed)
    items = pairs[:]
    rng.shuffle(items)
    n = len(items)
    n_tr = int(n * train)
    n_va = int(n * val)
    train_set = items[:n_tr]
    val_set   = items[n_tr:n_tr+n_va]
    test_set  = items[n_tr+n_va:]
    return {"train": train_set, "val": val_set, "test": test_set}

def merge_like_banknotes(img_sources, mask_sources, out_root, train=0.7, val=0.15, test=0.15, seed=1337, require_mask=False):
    out_root = Path(out_root)
    for split in ("train", "val", "test"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "masks").mkdir(parents=True, exist_ok=True)

    pairs = make_pairs(img_sources, mask_sources)
    if require_mask:
        pairs = [(i, m) for (i, m) in pairs if m is not None]
    if not pairs:
        raise SystemExit("❌ Không có cặp hợp lệ. Kiểm tra nguồn ảnh/mask và tên file (stem) trùng nhau.")

    splits = split_pairs(pairs, train=train, val=val, test=test, seed=seed)
    counts = {"train":0, "val":0, "test":0}

    for split, items in splits.items():
        img_dst = out_root / split / "images"
        msk_dst = out_root / split / "masks"
        for img, msk in items:
            # ảnh
            placed_img = copy_with_rename(img, img_dst)
            # mask (nếu có)
            if msk:
                # Đổi tên mask cho khớp stem cuối cùng của ảnh đã copy
                new_stem = placed_img.stem  # stem có thể đã thêm _001
                m_ext = msk.suffix.lower()
                placed_m = msk_dst / f"{new_stem}{m_ext}"
                if placed_m.exists():
                    # nếu đã tồn tại (hiếm), thì copy với hậu tố
                    copy_with_rename(msk, msk_dst)
                else:
                    shutil.copy2(msk, placed_m)
            counts[split] += 1

    print("✅ Done.")
    print(f"Train: {counts['train']} | Val: {counts['val']} | Test: {counts['test']}")
    print("Output at:", out_root.resolve())

if __name__ == "__main__":
    # ======= ĐIỀN SẴN ĐƯỜNG DẪN CỦA BẠN Ở ĐÂY =======
    # Nguồn ảnh có thể là nhiều thư mục khác nhau
    IMAGE_SOURCES = [
        r"C:\Users\Lazycat\Documents\AI\Number recognition\palmistry\Hands",                 # nguồn 1                                   # nguồn 2 (ví dụ)
    ]
    # Nguồn mask (nếu có) — có thể để trống []
    MASK_SOURCES = [
        r"C:\Users\Lazycat\Documents\AI\Number recognition\palmistry\skin_masks",            # masks tương ứng
    ]
    OUT_DIR = r"C:\Users\Lazycat\Documents\AI\Number recognition\palmistry\data"             # thư mục đích

    # Tỉ lệ chia & tuỳ chọn
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
    SEED = 1337
    REQUIRE_MASK = False  # True -> chỉ nhận mẫu có mask

    merge_like_banknotes(
        img_sources=IMAGE_SOURCES,
        mask_sources=MASK_SOURCES,
        out_root=OUT_DIR,
        train=TRAIN_RATIO, val=VAL_RATIO, test=TEST_RATIO,
        seed=SEED, require_mask=REQUIRE_MASK
    )
