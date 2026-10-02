import json
from pathlib import Path

from .grid_utils import normalize_grid


def normalize_task(task):
    normalized = {"train": [], "test": []}

    for sample in task.get("train", []):
        if "input" not in sample or "output" not in sample:
            continue
        inp = normalize_grid(sample["input"])
        out = normalize_grid(sample["output"])
        normalized["train"].append((inp, out))

    for sample in task.get("test", []):
        if "input" not in sample:
            continue
        normalized["test"].append({"input": normalize_grid(sample["input"])})

    return normalized


def load_task(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return normalize_task(data)


def load_tasks_from_directory(directory):
    tasks = []
    for path in sorted(Path(directory).glob("*.json")):
        tasks.append((path.name, load_task(path)))
    return tasks
