from .rule_library import choose_best_rule
from .grid_utils import normalize_grid


def solve_task(task):
    train_pairs = task.get("train", [])
    test_examples = task.get("test", [])

    if not train_pairs:
        return None

    ranked = choose_best_rule(task)
    if not ranked:
        return None

    best_rule = ranked[0]
    outputs = []
    for sample in test_examples:
        inp = normalize_grid(sample["input"])
        outputs.append(best_rule.apply(inp))

    if len(outputs) == 1:
        return outputs[0]
    return outputs
