from arc_solver.rule_library import RuleHypothesis, choose_best_rule


def test_translation_rule():
    task = {
        "train": [
            ([[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 2, 0]]),
        ],
        "test": [{"input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]]}],
    }
    ranked = choose_best_rule(task)
    assert ranked and ranked[0].name == "translation"


def test_rotation_rule():
    inp = [[1, 0], [0, 0]]
    out = [[0, 0], [0, 1]]
    rule = RuleHypothesis("rotation_90")
    assert rule.apply(inp) == out


def test_tile_repetition_rule():
    inp = [[1, 2], [2, 1]]
    rule = RuleHypothesis("tile_repetition", {"tile_h": 2, "tile_w": 2})
    assert rule.apply(inp) == inp


def test_diagonal_shift_rule():
    inp = [[1, 0], [0, 0]]
    rule = RuleHypothesis("diagonal_shift", {"offset_r": 1, "offset_c": 1})
    expected = [[0, 0], [0, 1]]
    assert rule.apply(inp) == expected


def test_recolor_by_frequency_rule():
    inp = [[1, 2], [1, 0]]
    rule = RuleHypothesis("recolor_by_frequency")
    out = rule.apply(inp)
    assert out[0][0] == 1 and out[0][1] == 2
