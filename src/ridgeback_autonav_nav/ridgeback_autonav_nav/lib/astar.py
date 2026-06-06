"""A* global planner over a float cost grid (pure numpy/heapq)."""
import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)


def plan(cost, start, goal, lethal_cost=254.0, allow_diagonal=True,
         max_expansions=400000):
    """Plan a path on a cost grid.

    Args:
        cost: (H, W) float array. Cells >= ``lethal_cost`` are impassable.
            Higher values are discouraged but traversable.
        start, goal: (col, row) integer cells.
        lethal_cost: threshold at/above which a cell blocks the path.
        allow_diagonal: 8-connectivity if True, else 4.
        max_expansions: safety cap.

    Returns:
        List of (col, row) cells from start to goal inclusive, or [] if no path.
    """
    h, w = cost.shape
    sc, sr = int(start[0]), int(start[1])
    gc, gr = int(goal[0]), int(goal[1])
    if not (0 <= sc < w and 0 <= sr < h and 0 <= gc < w and 0 <= gr < h):
        return []
    if cost[gr, gc] >= lethal_cost:
        return []

    if allow_diagonal:
        nbrs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))
    else:
        nbrs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0))

    def heuristic(c, r):
        dx, dy = abs(c - gc), abs(r - gr)
        if allow_diagonal:
            return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)
        return dx + dy

    start_idx = sr * w + sc
    goal_idx = gr * w + gc
    g_score = {start_idx: 0.0}
    came_from = {}
    open_heap = [(heuristic(sc, sr), start_idx)]
    closed = np.zeros(h * w, dtype=bool)
    expansions = 0

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if closed[cur]:
            continue
        if cur == goal_idx:
            return _reconstruct(came_from, cur, w)
        closed[cur] = True
        expansions += 1
        if expansions > max_expansions:
            return []
        cr, cc = divmod(cur, w)
        cur_g = g_score[cur]
        for dc, dr, step in nbrs:
            nc, nr = cc + dc, cr + dr
            if not (0 <= nc < w and 0 <= nr < h):
                continue
            cell_cost = cost[nr, nc]
            if cell_cost >= lethal_cost:
                continue
            nidx = nr * w + nc
            if closed[nidx]:
                continue
            # Step cost = distance + a fraction of the cell's inflation cost.
            tentative = cur_g + step * (1.0 + cell_cost / 50.0)
            if tentative < g_score.get(nidx, math.inf):
                g_score[nidx] = tentative
                came_from[nidx] = cur
                heapq.heappush(open_heap, (tentative + heuristic(nc, nr), nidx))
    return []


def _reconstruct(came_from, cur, w):
    path = []
    while cur in came_from:
        r, c = divmod(cur, w)
        path.append((c, r))
        cur = came_from[cur]
    r, c = divmod(cur, w)
    path.append((c, r))
    path.reverse()
    return path


def nearest_free_cell(cost, cell, lethal_cost=254.0, max_radius=40):
    """Spiral-search the nearest non-lethal cell to ``cell`` (for goal snapping)."""
    h, w = cost.shape
    c0, r0 = int(cell[0]), int(cell[1])
    if 0 <= c0 < w and 0 <= r0 < h and cost[r0, c0] < lethal_cost:
        return (c0, r0)
    for radius in range(1, max_radius + 1):
        best = None
        best_d = None
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                c, r = c0 + dc, r0 + dr
                if 0 <= c < w and 0 <= r < h and cost[r, c] < lethal_cost:
                    d = dc * dc + dr * dr
                    if best_d is None or d < best_d:
                        best_d, best = d, (c, r)
        if best is not None:
            return best
    return None


def downsample_path(path, every=3):
    """Keep every Nth cell plus the endpoints (lighter to publish/follow)."""
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    out += [path[i] for i in range(every, len(path) - 1, every)]
    out.append(path[-1])
    return out
