#!/usr/bin/env python3
"""
Create a square-cropped copy of a LeRobot-format GR00T dataset.

Everything (meta/, data/ parquet, structure) is copied verbatim; only the videos
under videos/**/*.mp4 are re-encoded with a centered square crop (crop to the
shorter edge). meta/info.json video `shape` fields are rewritten to the new
square resolution. Frame count and fps are preserved so the parquet frame
indices stay aligned.

Usage:
    python crop_dataset_square.py \
        --src /home/mz/Documents/small_targets/g1_dataset_small_finecontroller \
        --dst /home/mz/Documents/small_targets/g1_dataset_small_finecontroller_square \
        --jobs 4

Rationale: the GR00T resize pipeline letterboxes 16:9 frames to a square, wasting
~25% of the visual-token budget on black bars and shrinking small targets. Feeding
already-square frames removes that waste (see resolution_diag/).
"""
import argparse
import concurrent.futures as cf
import json
import shutil
import subprocess
from pathlib import Path


def probe(path: Path, entries: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", f"stream={entries}", "-of", "default=noprint_wrappers=1:nokey=0",
         str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    d = {}
    for line in out.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


def crop_one(src: Path, dst: Path, crf: int) -> tuple[Path, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # crop to a centered square of side = min(width, height)
    vf = "crop='min(iw,ih)':'min(iw,ih)':'(iw-min(iw,ih))/2':'(ih-min(iw,ih))/2'"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    # verify frame-count parity (crop must not drop/dup frames)
    ns = probe(src, "nb_frames").get("nb_frames")
    nd = probe(dst, "nb_frames").get("nb_frames")
    status = "OK" if ns == nd else f"FRAME-COUNT MISMATCH src={ns} dst={nd}"
    return dst, status


def copy_non_video(src_root: Path, dst_root: Path):
    for item in src_root.iterdir():
        if item.name == "videos":
            continue
        target = dst_root / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def rewrite_info_json(dst_root: Path):
    info_path = dst_root / "meta" / "info.json"
    if not info_path.exists():
        return
    info = json.loads(info_path.read_text())
    changed = []
    for key, feat in info.get("features", {}).items():
        if feat.get("dtype") == "video" and "shape" in feat:
            names = feat.get("names", [])
            shape = list(feat["shape"])
            try:
                hi, wi = names.index("height"), names.index("width")
            except ValueError:
                # assume [C, H, W]
                hi, wi = 1, 2
            side = min(shape[hi], shape[wi])
            if shape[hi] != side or shape[wi] != side:
                shape[hi] = shape[wi] = side
                feat["shape"] = shape
                changed.append(f"{key} -> {shape}")
    info_path.write_text(json.dumps(info, indent=4))
    print("[info.json] updated:", ", ".join(changed) if changed else "no video shapes changed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--crf", type=int, default=16, help="libx264 quality (lower=better, 16≈visually lossless)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only process N videos (for testing)")
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    assert src.exists(), f"source not found: {src}"
    assert src != dst, "src and dst must differ"

    videos = sorted((src / "videos").rglob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    print(f"Cropping {len(videos)} videos: {src.name} -> {dst.name}  (crf={args.crf}, jobs={args.jobs})")

    dst.mkdir(parents=True, exist_ok=True)
    if not args.limit:
        print("Copying meta/ + data/ (verbatim)...")
        copy_non_video(src, dst)
        rewrite_info_json(dst)

    tasks = [(v, dst / v.relative_to(src)) for v in videos]
    n_ok = 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(crop_one, s, d, args.crf): d for s, d in tasks}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            d, status = fut.result()
            n_ok += status == "OK"
            flag = "" if status == "OK" else f"  <<< {status}"
            print(f"[{i}/{len(tasks)}] {d.relative_to(dst)}  {status}{flag}")
    print(f"\nDone: {n_ok}/{len(tasks)} videos OK. New dataset at {dst}")


if __name__ == "__main__":
    main()
