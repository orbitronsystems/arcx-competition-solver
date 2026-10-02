from collections import defaultdict, deque


def normalize_grid(grid):
    return [list(map(int, row)) for row in grid]


def clone_grid(grid):
    return [row[:] for row in grid]


def empty_grid(h, w, fill=0):
    return [[fill for _ in range(w)] for _ in range(h)]


def grid_shape(grid):
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]))


def grids_equal(a, b):
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    if len(a[0]) != len(b[0]):
        return False
    return a == b


def color_set(grid):
    colors = set()
    for row in grid:
        for v in row:
            colors.add(v)
    return colors


def bbox_of_nonzero(grid):
    coords = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0]
    if not coords:
        return (0, 0, -1, -1)
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (min(rs), min(cs), max(rs), max(cs))


def crop_nonzero(grid):
    r1, c1, r2, c2 = bbox_of_nonzero(grid)
    if r1 == 0 and c1 == 0 and r2 == -1 and c2 == -1:
        return []
    return [row[c1:c2 + 1] for row in grid[r1:r2 + 1]]


def normalize_shape(grid):
    cropped = crop_nonzero(grid)
    if not cropped:
        return [[0]]
    return cropped


def pad_grid(grid, h, w, fill=0):
    out = [[fill for _ in range(w)] for _ in range(h)]
    gh, gw = len(grid), len(grid[0]) if grid else 0
    for r in range(gh):
        for c in range(gw):
            out[r][c] = grid[r][c]
    return out


def flood_fill(grid, sr, sc, target, visited=None):
    h, w = len(grid), len(grid[0])
    if visited is None:
        visited = set()
    q = deque([(sr, sc)])
    visited.add((sr, sc))
    cells = []
    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in visited and grid[rr][cc] == target:
                visited.add((rr, cc))
                q.append((rr, cc))
    return cells


def connected_components(grid):
    h, w = len(grid), len(grid[0]) if grid else 0
    seen = set()
    comps = []

    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            color = grid[r][c]
            if color == 0:
                seen.add((r, c))
                continue

            cells = flood_fill(grid, r, c, color, seen)
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            bbox = (min(rs), min(cs), max(rs), max(cs))
            comps.append({
                "color": color,
                "cells": cells,
                "bbox": bbox,
                "area": len(cells),
                "center": (
                    (min(rs) + max(rs)) / 2.0,
                    (min(cs) + max(cs)) / 2.0
                ),
            })

            for p in cells:
                seen.add(p)

    return comps


def color_hist(grid):
    hist = defaultdict(int)
    for row in grid:
        for v in row:
            hist[v] += 1
    return dict(hist)


def rotate90(grid):
    h, w = len(grid), len(grid[0])
    return [[grid[h - 1 - r][c] for r in range(h)] for c in range(w)]


def rotate180(grid):
    return [list(reversed(row)) for row in reversed(grid)]


def reflect_h(grid):
    return [list(reversed(row)) for row in grid]


def reflect_v(grid):
    return list(reversed(grid))
