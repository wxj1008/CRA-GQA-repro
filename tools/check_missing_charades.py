import os
from pathlib import Path
import json

# ===== 配置路径（按你的项目结构）=====
ROOT = Path(__file__).resolve().parents[1]
CHARADES_DIR = ROOT / "data/star/Charades_v1_480"

# STAR 的 annotation（通常是 json）
ANNOTATION_FILES = [
    ROOT / "data/star/STAR_train.json",
    ROOT / "data/star/STAR_val.json",
    ROOT / "data/star/STAR_test.json",
]

# ===== 收集所有需要的视频 id =====
video_ids = set()

for ann_path in ANNOTATION_FILES:
    if not ann_path.exists():
        print(f"[WARN] annotation not found: {ann_path}")
        continue

    with open(ann_path, "r") as f:
        data = json.load(f)

    # 兼容 STAR 常见格式
    for item in data:
        if "video_id" in item:
            video_ids.add(item["video_id"])
        elif "vid" in item:
            video_ids.add(item["vid"])

print(f"[INFO] total referenced videos: {len(video_ids)}")

# ===== 检查文件是否存在 =====
missing = []

for vid in sorted(video_ids):
    mp4_path = CHARADES_DIR / f"{vid}.mp4"
    if not mp4_path.exists():
        missing.append(vid)

print(f"[INFO] missing videos: {len(missing)}")

# ===== 写出结果 =====
out_path = ROOT / "missing_charades_videos.txt"
with open(out_path, "w") as f:
    for vid in missing:
        f.write(vid + "\n")

print(f"[DONE] missing list written to: {out_path}")
