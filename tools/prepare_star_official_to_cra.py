import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

STAR_DIR = "data/star"
MAX_FEATS = 32


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def normalize_item(item):
    # question id
    qid = str(item.get("question_id", item.get("qid", item.get("id"))))

    # video id
    vid = str(item.get("video_id", item.get("video", item.get("vid"))))

    # question
    question = item.get("question", item.get("query", ""))

    # question type
    qtype = str(item.get("type", item.get("question_type", "null")))

    # options
    options = item.get("choices", item.get("options", item.get("candidates")))
    if options is None or len(options) < 4:
        raise ValueError(f"Invalid options for qid={qid}")
    options = [str(x) for x in options[:4]]

    # answer
    ans = item.get("answer")
    if isinstance(ans, int):
        answer = options[ans]
    elif isinstance(ans, str) and ans in ["A", "B", "C", "D"]:
        answer = options[["A", "B", "C", "D"].index(ans)]
    else:
        answer = str(ans)

    return {
        "video_id": vid,
        "qid": qid,
        "question": question,
        "type": qtype,
        "answer": answer,
        "a0": options[0],
        "a1": options[1],
        "a2": options[2],
        "a3": options[3],
    }


def build_csv(split):
    src = os.path.join(STAR_DIR, f"STAR_{split}.json")
    dst = os.path.join(STAR_DIR, f"{split}.csv")

    data = load_json(src)
    rows = [normalize_item(x) for x in tqdm(data, desc=f"CSV {split}")]
    df = pd.DataFrame(rows)
    df.to_csv(dst, index=False)
    print(f"[OK] {dst}  rows={len(df)}")


def build_gsub_and_frame2time(split):
    seg_csv = os.path.join(STAR_DIR, "Video_Segments.csv")
    qa_csv = os.path.join(STAR_DIR, f"{split}.csv")

    segs = pd.read_csv(seg_csv)
    qa = pd.read_csv(qa_csv, keep_default_na=False)

    segs["question_id"] = segs["question_id"].astype(str)
    seg_map = {
        qid: (float(s), float(e))
        for qid, s, e in zip(segs["question_id"], segs["start"], segs["end"])
    }

    gsub = {}
    frame2time = {}

    for vid, g in qa.groupby("video_id"):
        vid = str(vid)
        loc = {}
        ends = []

        for qid in g["qid"].astype(str).tolist():
            if qid in seg_map:
                s, e = seg_map[qid]
            else:
                s, e = 0.0, 0.0
            loc[qid] = [[s, e]]
            ends.append(e)

        duration = max(ends) if ends else 0.0
        gsub[vid] = {"duration": duration, "fps": 0, "location": loc}

        if duration > 0:
            frame2time[vid] = np.linspace(0, duration, MAX_FEATS).round(2).tolist()
        else:
            frame2time[vid] = [0.0] * MAX_FEATS

    save_json(gsub, os.path.join(STAR_DIR, f"gsub_{split}.json"))
    save_json(frame2time, os.path.join(STAR_DIR, f"frame2time_{split}.json"))
    print(f"[OK] gsub / frame2time for {split}")


if __name__ == "__main__":
    os.makedirs(STAR_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        build_csv(split)
        build_gsub_and_frame2time(split)
