import json
from pathlib import Path

from .task_loader import load_tasks_from_directory
from .solver import solve_task


def run_directory(tasks_dir, output_dir=None):
    tasks_dir = Path(tasks_dir)
    if output_dir is None:
        output_dir = tasks_dir / "predictions"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task_name, task in load_tasks_from_directory(tasks_dir):
        prediction = solve_task(task)
        result = {
            "task": task_name,
            "prediction": prediction,
        }
        results.append(result)

        out_path = output_dir / f"{Path(task_name).stem}_prediction.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m arc_solver.benchmark_runner <tasks_dir>")
        sys.exit(1)

    tasks_dir = sys.argv[1]
    results = run_directory(tasks_dir)
    print(f"Processed {len(results)} tasks.")
