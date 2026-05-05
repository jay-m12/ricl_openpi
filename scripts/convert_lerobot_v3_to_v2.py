from pathlib import Path
import argparse
import json
import pandas as pd


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.readline().strip()


def main(dataset_root: str):
    root = Path(dataset_root).expanduser().resolve()
    meta = root / "meta"

    if not meta.exists():
        raise FileNotFoundError(f"meta directory not found: {meta}")

    tasks_parquet = meta / "tasks.parquet"
    episodes_dir = meta / "episodes"
    info_json = meta / "info.json"

    if not tasks_parquet.exists():
        raise FileNotFoundError(f"tasks.parquet not found: {tasks_parquet}")
    if not episodes_dir.exists():
        raise FileNotFoundError(f"episodes directory not found: {episodes_dir}")
    if not info_json.exists():
        raise FileNotFoundError(f"info.json not found: {info_json}")

    tasks_df = pd.read_parquet(tasks_parquet).reset_index()

    if "task" not in tasks_df.columns:
        raise KeyError(f"'task' column not found after reset_index(). columns={tasks_df.columns.tolist()}")
    if "task_index" not in tasks_df.columns:
        raise KeyError(f"'task_index' column not found. columns={tasks_df.columns.tolist()}")

    tasks_jsonl = meta / "tasks.jsonl"
    with tasks_jsonl.open("w", encoding="utf-8") as f:
        for _, row in tasks_df.iterrows():
            f.write(json.dumps({
                "task_index": int(row["task_index"]),
                "task": str(row["task"]),
            }, ensure_ascii=False) + "\n")

    canonical_task = str(tasks_df.iloc[0]["task"])

    episode_files = sorted(episodes_dir.glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode parquet files found under: {episodes_dir}")

    episodes_jsonl = meta / "episodes.jsonl"
    with episodes_jsonl.open("w", encoding="utf-8") as f:
        for p in episode_files:
            ep_df = pd.read_parquet(p)
            required = {"episode_index", "length"}
            missing = required - set(ep_df.columns)
            if missing:
                raise KeyError(f"{p} missing columns: {missing}")

            for _, row in ep_df.iterrows():
                f.write(json.dumps({
                    "episode_index": int(row["episode_index"]),
                    "tasks": [canonical_task],
                    "length": int(row["length"]),
                }, ensure_ascii=False) + "\n")

    info = json.loads(info_json.read_text(encoding="utf-8"))

    backup = meta / "info.json.v3_backup"
    if not backup.exists():
        backup.write_text(json.dumps(info, indent=2), encoding="utf-8")

    for key in ["data_path", "video_path"]:
        if key in info and isinstance(info[key], str):
            info[key] = info[key].replace("{chunk_index:03d}", "{episode_chunk:03d}")

    info_json.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("Done.")
    print("tasks.jsonl lines:", count_lines(tasks_jsonl))
    print("episodes.jsonl lines:", count_lines(episodes_jsonl))
    print("first tasks.jsonl:", first_line(tasks_jsonl))
    print("first episodes.jsonl:", first_line(episodes_jsonl))
    print("data_path:", info.get("data_path"))
    print("video_path:", info.get("video_path"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    args = parser.parse_args()
    main(args.dataset_root)