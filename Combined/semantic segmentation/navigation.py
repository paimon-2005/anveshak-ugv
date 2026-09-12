import heapq
import math

import cv2
import numpy as np


def nearest_free_cell(free_grid, preferred_cell, maximum_distance=None):
    cells = np.argwhere(free_grid)
    if not len(cells):
        return None
    distances = np.sum((cells - np.asarray(preferred_cell)) ** 2, axis=1)
    index = int(np.argmin(distances))
    if maximum_distance is not None and distances[index] > maximum_distance ** 2:
        return None
    return tuple(map(int, cells[index]))


def astar_search(free_grid, start, goal):
    grid = np.asarray(free_grid, bool)
    rows, columns = grid.shape
    for cell in (start, goal):
        if cell is None or not (0 <= cell[0] < rows and 0 <= cell[1] < columns) or not grid[cell]:
            return []
    diagonal = math.sqrt(2)
    neighbours = ((-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
                  (-1, -1, diagonal), (-1, 1, diagonal), (1, -1, diagonal), (1, 1, diagonal))

    def heuristic(cell):
        y, x = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        return max(y, x) + (diagonal - 1) * min(y, x)

    costs = np.full(grid.shape, np.inf)
    costs[start] = 0
    parents = {}
    pending = [(heuristic(start), 0.0, start)]
    while pending:
        _, cost, cell = heapq.heappop(pending)
        if cost > costs[cell]:
            continue
        if cell == goal:
            path = [cell]
            while cell in parents:
                cell = parents[cell]
                path.append(cell)
            return path[::-1]
        row, col = cell
        for dy, dx, step in neighbours:
            other = row + dy, col + dx
            if not (0 <= other[0] < rows and 0 <= other[1] < columns and grid[other]):
                continue
            if dy and dx and not (grid[row + dy, col] and grid[row, col + dx]):
                continue
            candidate = cost + step
            if candidate < costs[other]:
                costs[other] = candidate
                parents[other] = cell
                heapq.heappush(pending, (candidate + heuristic(other), candidate, other))
    return []


def plan_shortest_path(traversable_mask, cell_size=8, clearance=16, goal_fraction=0.15):
    mask = np.asarray(traversable_mask)
    if mask.ndim != 2 or cell_size < 1 or clearance < 0 or not 0 <= goal_fraction < 1:
        raise ValueError("Invalid planner inputs")
    h, w = mask.shape
    rows, cols = h // cell_size, w // cell_size
    if min(rows, cols) < 2:
        return [], "IMAGE TOO SMALL"
    safe = (mask > 0).astype(np.uint8)
    if clearance:
        radius = math.ceil(clearance)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        padded = cv2.copyMakeBorder(safe, radius, radius, radius, radius, cv2.BORDER_REPLICATE)
        safe = cv2.erode(padded, kernel)[radius:-radius, radius:-radius]
        safe[:, :radius] = 0
        safe[:, -radius:] = 0
    free = safe[:rows * cell_size, :cols * cell_size].reshape(rows, cell_size, cols, cell_size).all(axis=(1, 3))
    start = rows - 1, cols // 2
    if not free[start]:
        return [], "START BLOCKED"
    _, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    reachable = np.argwhere(labels == labels[start])
    order = np.lexsort((np.abs(reachable[:, 1] - cols // 2), np.abs(reachable[:, 0] - int(rows * goal_fraction))))
    goal = tuple(map(int, reachable[order[0]]))
    if start[0] - goal[0] < max(2, rows // 10):
        return [], "NO FORWARD PATH"
    path = astar_search(free, start, goal)
    return ([(int((x + 0.5) * cell_size), int((y + 0.5) * cell_size)) for y, x in path],
            "PATH FOUND" if path else "NO PATH")


def draw_planned_path(image, path):
    if not path:
        return
    if len(path) > 1:
        cv2.polylines(image, [np.asarray(path, np.int32).reshape(-1, 1, 2)], False, (255, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(image, path[0], 5, (0, 255, 255), -1)
    cv2.circle(image, path[-1], 5, (0, 0, 255), -1)
