import os
import json
import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import open_clip
from PIL import Image


STAR_DIR = "data/star"
VIDEO_DIR = os.path.join(STAR_DIR, "Charades_v1_480")
OUT_DIR = os.path.join(STAR_DIR, "video_feature", "CLIP_L")
MAX_FEATS = 32


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def uniform_ids(nf, k=MAX_FEATS):
    if nf <= 0:
        return [0] * k
    return np.linspace(0, nf - 1, k).round().astype(int).tolist()


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device).eval()

    os.makedirs(OUT_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(STAR_DIR, f"{split}.csv"))
        vids = sorted(df["video_id"].astype(str).unique().tolist())

        gsub = load_json(os.path.join(STAR_DIR, f"gsub_{split}.json"))
        frame2time = load_json(os.path.join(STAR_DIR, f"frame2time_{split}.json"))

        feats_all = []
        vids_bytes = []

        for vid in tqdm(vids, desc=f"CLIP-L {split}"):
            path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
            if not os.path.exists(path):
                raise FileNotFoundError(path)

            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = nf / fps if fps and fps > 0 else 0.0

            ids = uniform_ids(nf)
            ts = [(i / fps) if fps and fps > 0 else 0.0 for i in ids]

            frames = []
            for i in ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            imgs = [preprocess(Image.fromarray(f)) for f in frames]
            imgs = torch.stack(imgs, dim=0).to(device)
            imgs = imgs.to(dtype=next(model.parameters()).dtype)


            emb = model.encode_image(imgs)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            emb = emb.cpu().numpy().astype("float32")

            feats_all.append(emb)
            vids_bytes.append(vid.encode())

            gsub[vid]["duration"] = float(duration)
            gsub[vid]["fps"] = float(fps) if fps else 0.0
            frame2time[vid] = [round(t, 2) for t in ts]

        feats_all = np.stack(feats_all)

        out_h5 = os.path.join(OUT_DIR, f"{split}.h5")
        with h5py.File(out_h5, "w") as f:
            f.create_dataset("vid", data=np.array(vids_bytes, dtype="S"))
            f.create_dataset("CLIPL_I", data=feats_all)

        save_json(gsub, os.path.join(STAR_DIR, f"gsub_{split}.json"))
        save_json(frame2time, os.path.join(STAR_DIR, f"frame2time_{split}.json"))

        print(f"[OK] {out_h5}  shape={feats_all.shape}")


if __name__ == "__main__":
    main()
