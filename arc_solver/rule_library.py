from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .grid_utils import (
    bbox_of_nonzero,
    clone_grid,
    color_set,
    connected_components,
    crop_nonzero,
    normalize_shape,
    pad_grid,
    reflect_h,
    reflect_v,
    rotate180,
    rotate90,
)


@dataclass
class RuleHypothesis:
    name: str
    params: dict = field(default_factory=dict)
    score: float = 0.0

    def apply(self, grid):
        if self.name == "translation":
            return self._apply_translation(grid)

        if self.name == "rotation_90":
            return rotate90(grid)

        if self.name == "rotation_180":
            return rotate180(grid)

        if self.name == "reflection_h":
            return reflect_h(grid)

        if self.name == "reflection_v":
            return reflect_v(grid)

        if self.name == "recolor_by_rank":
            return self._apply_recolor_by_rank(grid)

        if self.name == "recolor_by_frequency":
            return self._apply_recolor_by_frequency(grid)

        if self.name == "crop_pad":
            return pad_grid(crop_nonzero(grid), len(grid), len(grid[0]))

        if self.name == "fill_bbox":
            return self._apply_fill_bbox(grid)

        if self.name == "object_count_to_color":
            return self._apply_object_count_to_color(grid)

        if self.name == "tile_repetition":
            return self._apply_tile_repetition(grid)

        if self.name == "bounding_box_extract":
            return self._apply_bounding_box_extract(grid)

        if self.name == "bounding_box_fill_sparse":
            return self._apply_bounding_box_fill_sparse(grid)

        if self.name == "diagonal_shift":
            return self._apply_diagonal_shift(grid)

        if self.name == "pattern_completion":
            return self._apply_pattern_completion(grid)

        if self.name == "no_op":
            return [row[:] for row in grid]

        raise ValueError(f"Unknown rule: {self.name}")

    def _apply_translation(self, grid):
        dx = self.params.get("dx", 0)
        dy = self.params.get("dy", 0)
        h, w = len(grid), len(grid[0])
        out = [[0 for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if grid[r][c] != 0:
                    rr = r + dy
                    cc = c + dx
                    if 0 <= rr < h and 0 <= cc < w:
                        out[rr][cc] = grid[r][c]
        return out

    def _apply_recolor_by_rank(self, grid):
        unique = sorted({v for row in grid for v in row if v != 0})
        mapping = {old: i + 1 for i, old in enumerate(unique)}
        h, w = len(grid), len(grid[0])
        out = [[0 for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                v = grid[r][c]
                if v != 0:
                    out[r][c] = mapping[v]
        return out

    def _apply_recolor_by_frequency(self, grid):
        freq = Counter(v for row in grid for v in row if v != 0)
        if not freq:
            return grid

        sorted_colors = [color for color, _ in freq.most_common()]
        mapping = {old: i + 1 for i, old in enumerate(sorted_colors)}
        h, w = len(grid), len(grid[0])
        out = [[0 for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                v = grid[r][c]
                if v != 0:
                    out[r][c] = mapping[v]
        return out

    def _apply_fill_bbox(self, grid):
        r1, c1, r2, c2 = bbox_of_nonzero(grid)
        if r1 == 0 and c1 == 0 and r2 == -1 and c2 == -1:
            return grid

        color = grid[r1][c1]
        out = [row[:] for row in grid]
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                out[r][c] = color
        return out

    def _apply_object_count_to_color(self, grid):
        components = connected_components(grid)
        h, w = len(grid), len(grid[0])
        out = [[0 for _ in range(w)] for _ in range(h)]

        color_to_components = defaultdict(list)
        for comp in components:
            color_to_components[comp["color"]].append(comp)

        for comp in components:
            count = len(color_to_components[comp["color"]])
            for r, c in comp["cells"]:
                out[r][c] = count

        return out

    def _apply_tile_repetition(self, grid):
        tile_h = self.params.get("tile_h", 2)
        tile_w = self.params.get("tile_w", 2)

        h, w = len(grid), len(grid[0])
        base_tile = []
        for r in range(min(tile_h, h)):
            row = []
            for c in range(min(tile_w, w)):
                row.append(grid[r][c])
            base_tile.append(row)

        if len(base_tile) < tile_h:
            base_tile.extend([[0] * tile_w for _ in range(tile_h - len(base_tile))])
        for r in range(len(base_tile)):
            if len(base_tile[r]) < tile_w:
                base_tile[r].extend([0] * (tile_w - len(base_tile[r])))

        out = []
        for r in range(h):
            row = []
            for c in range(w):
                tr = r % tile_h
                tc = c % tile_w
                row.append(base_tile[tr][tc])
            out.append(row)

        return out

    def _apply_bounding_box_extract(self, grid):
        return crop_nonzero(grid)

    def _apply_bounding_box_fill_sparse(self, grid):
        components = connected_components(grid)
        out = clone_grid(grid)

        for comp in components:
            r1, c1, r2, c2 = comp["bbox"]
            color = comp["color"]
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if out[r][c] == 0:
                        out[r][c] = color

        return out

    def _apply_diagonal_shift(self, grid):
        offset_r = self.params.get("offset_r", 1)
        offset_c = self.params.get("offset_c", 1)

        h, w = len(grid), len(grid[0])
        out = [[0 for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if grid[r][c] != 0:
                    rr = r + offset_r
                    cc = c + offset_c
                    if 0 <= rr < h and 0 <= cc < w:
                        out[rr][cc] = grid[r][c]
        return out

    def _apply_pattern_completion(self, grid):
        h, w = len(grid), len(grid[0])
        out = clone_grid(grid)

        r1, c1, r2, c2 = bbox_of_nonzero(grid)
        if r1 == 0 and c1 == 0 and r2 == -1 and c2 == -1:
            return grid

        bbox_h = r2 - r1 + 1
        bbox_w = c2 - c1 + 1

        for period in range(1, min(bbox_h, bbox_w) + 1):
            if bbox_h % period == 0 or bbox_w % period == 0:
                template = []
                for r in range(r1, min(r1 + period, r2 + 1)):
                    row = []
                    for c in range(c1, min(c1 + period, c2 + 1)):
                        row.append(grid[r][c])
                    template.append(row)

                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        tr = (r - r1) % len(template)
                        tc = (c - c1) % len(template[0]) if template else 0
                        if tr < len(template) and tc < len(template[tr]):
                            out[r][c] = template[tr][tc]
        return out


def infer_translation(inp, out):
    h, w = len(inp), len(inp[0])
    inp_coords = [(r, c) for r in range(h) for c in range(w) if inp[r][c] != 0]

    if not inp_coords:
        return None

    i_bbox = bbox_of_nonzero(inp)
    o_bbox = bbox_of_nonzero(out)
    if i_bbox == (0, 0, -1, -1) or o_bbox == (0, 0, -1, -1):
        return None

    dy = o_bbox[0] - i_bbox[0]
    dx = o_bbox[1] - i_bbox[1]

    for (r, c) in inp_coords:
        rr = r + dy
        cc = c + dx
        if rr < 0 or cc < 0 or rr >= len(out) or cc >= len(out[0]):
            return None
        if out[rr][cc] != inp[r][c]:
            return None

    return RuleHypothesis("translation", {"dx": dx, "dy": dy})


def infer_simple_symmetry(inp, out):
    candidates = []
    transforms = [
        ("rotation_90", rotate90),
        ("rotation_180", rotate180),
        ("reflection_h", reflect_h),
        ("reflection_v", reflect_v),
    ]

    for name, fn in transforms:
        if fn(inp) == out:
            candidates.append(RuleHypothesis(name))
    return candidates


def infer_recolor_by_rank(inp, out):
    if color_set(inp) == color_set(out):
        return RuleHypothesis("recolor_by_rank")

    unique_in = sorted({v for row in inp for v in row if v != 0})
    unique_out = sorted({v for row in out for v in row if v != 0})
    if len(unique_in) == len(unique_out) and len(unique_in) > 0:
        return RuleHypothesis("recolor_by_rank")
    return None


def infer_recolor_by_frequency(inp, out):
    freq_in = Counter(v for row in inp for v in row if v != 0)
    freq_out = Counter(v for row in out for v in row if v != 0)
    if len(freq_in) == len(freq_out) and len(freq_in) > 0:
        return RuleHypothesis("recolor_by_frequency")
    return None


def infer_crop_pad(inp, out):
    if normalize_shape(inp) == normalize_shape(out):
        return RuleHypothesis("crop_pad")
    return None


def infer_fill_bbox(inp, out):
    if bbox_of_nonzero(inp) != bbox_of_nonzero(out):
        return RuleHypothesis("fill_bbox")
    return None


def infer_object_count(inp, out):
    components_in = connected_components(inp)
    if len(components_in) < 2:
        return None

    h, w = len(inp), len(inp[0])
    color_count = defaultdict(int)
    for comp in components_in:
        color_count[comp["color"]] += 1

    expected = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if inp[r][c] == 0:
                continue
            for comp in components_in:
                if (r, c) in comp["cells"]:
                    expected[r][c] = color_count[comp["color"]]
                    break

    # Check if output values are object counts
    if expected == out:
        return RuleHypothesis("object_count_to_color")

    return None


def infer_tile_repetition(inp, out):
    h_in, w_in = len(inp), len(inp[0])
    h_out, w_out = len(out), len(out[0])

    for tile_h in range(1, min(h_in, 5)):
        for tile_w in range(1, min(w_in, 5)):
            match = True
            for r in range(h_out):
                for c in range(w_out):
                    tr = r % tile_h
                    tc = c % tile_w
                    if tr < h_in and tc < w_in and out[r][c] != inp[tr][tc]:
                        match = False
                        break
                if not match:
                    break
            if match:
                return RuleHypothesis("tile_repetition", {"tile_h": tile_h, "tile_w": tile_w})
    return None


def infer_bounding_box_extract(inp, out):
    if crop_nonzero(inp) == out:
        return RuleHypothesis("bounding_box_extract")
    return None


def infer_diagonal_shift(inp, out):
    h, w = len(inp), len(inp[0])
    coords = [(r, c) for r in range(h) for c in range(w) if inp[r][c] != 0]
    if not coords:
        return None

    r0, c0 = coords[0]
    for rr in range(-max(h, w), max(h, w) + 1):
        for cc in range(-max(h, w), max(h, w) + 1):
            if rr == 0 and cc == 0:
                continue
            ok = True
            for r, c in coords:
                nr = r + rr
                nc = c + cc
                if nr < 0 or nc < 0 or nr >= len(out) or nc >= len(out[0]):
                    ok = False
                    break
                if out[nr][nc] != inp[r][c]:
                    ok = False
                    break
            if ok:
                return RuleHypothesis("diagonal_shift", {"offset_r": rr, "offset_c": cc})
    return None


def infer_pattern_completion(inp, out):
    bbox_in = bbox_of_nonzero(inp)
    bbox_out = bbox_of_nonzero(out)
    if bbox_in == (0, 0, -1, -1):
        return None
    if bbox_in[2] - bbox_in[0] < bbox_out[2] - bbox_out[0]:
        return RuleHypothesis("pattern_completion")
    return None


def infer_bounding_box_fill_sparse(inp, out):
    comps_in = connected_components(inp)
    comps_out = connected_components(out)
    if len(comps_in) == len(comps_out):
        for cin, cout in zip(comps_in, comps_out):
            if cin["bbox"] == cout["bbox"] and cout["area"] > cin["area"]:
                return RuleHypothesis("bounding_box_fill_sparse")
    return None


def generate_rule_candidates(inp, out):
    rules = []

    for rule in [
        infer_translation(inp, out),
        infer_recolor_by_rank(inp, out),
        infer_recolor_by_frequency(inp, out),
        infer_crop_pad(inp, out),
        infer_fill_bbox(inp, out),
        infer_object_count(inp, out),
        infer_tile_repetition(inp, out),
        infer_bounding_box_extract(inp, out),
        infer_diagonal_shift(inp, out),
        infer_pattern_completion(inp, out),
        infer_bounding_box_fill_sparse(inp, out),
    ]:
        if rule is not None:
            rules.append(rule)

    for rule in infer_simple_symmetry(inp, out):
        rules.append(rule)

    rules.append(RuleHypothesis("no_op"))
    return rules


def score_rule_on_examples(train_pairs, rule):
    total = 0.0
    for inp, expected in train_pairs:
        try:
            pred = rule.apply(inp)
        except Exception:
            return 0.0

        if pred == expected:
            total += 1.0
            continue

        h = len(expected)
        w = len(expected[0]) if expected else 0
        cells = max(1, h * w)
        matches = 0
        for r in range(h):
            for c in range(w):
                if r < len(pred) and c < len(pred[r]) and pred[r][c] == expected[r][c]:
                    matches += 1
        total += matches / cells

    return total


def choose_best_rule(task):
    train_pairs = task.get("train", [])
    if not train_pairs:
        return []

    rules = []
    for inp, out in train_pairs:
        for rule in generate_rule_candidates(inp, out):
            rules.append(rule)

    dedup = {}
    for rule in rules:
        key = (rule.name, tuple(sorted(rule.params.items())))
        dedup.setdefault(key, rule)

    unique_rules = list(dedup.values())
    scored = []
    for rule in unique_rules:
        rule.score = score_rule_on_examples(train_pairs, rule)
        scored.append(rule)

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:10]
