"""Detector result selection and conservative spatial association (no model calls)."""
import math


def select_targets(boxes, scores, command, width, height, threshold):
    """An empty/invalid result is a normal outcome, never a fallback target."""
    valid = []
    for box, score in zip(boxes, scores):
        values = [float(v) for v in box]
        score = float(score)
        if len(values) != 4 or not all(math.isfinite(v) for v in [*values, score]) or score < threshold:
            continue
        x1, y1, x2, y2 = values
        box = [max(0, min(width, x1)), max(0, min(height, y1)),
               max(0, min(width, x2)), max(0, min(height, y2))]
        if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
            valid.append({"box": box, "score": score})
    valid.sort(key=lambda d: (d["box"][0] + d["box"][2]) / 2)
    candidates = [dict(id=i + 1, **item) for i, item in enumerate(valid)]
    selected, reason = [], ""
    if not candidates:
        reason = "当前画面未找到满足阈值的目标，正在等待后续画面或新的描述。"
    elif command.quantity == "all":
        selected = candidates.copy()
    elif command.ordinal is not None:
        if command.ordinal > len(candidates):
            reason = f"本帧只找到 {len(candidates)} 个候选，无法选择 {command.ordinal} 号。"
        else:
            selected = [candidates[command.ordinal - 1]]
    else:
        area = lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
        if command.selector == "left":
            chosen = candidates[0]
        elif command.selector == "right":
            chosen = candidates[-1]
        elif command.selector == "largest":
            chosen = max(candidates, key=area)
        elif command.selector == "smallest":
            chosen = min(candidates, key=area)
        else:
            chosen = max(candidates, key=lambda d: d["score"])
        selected = [chosen]
    return {"matched": bool(selected), "reason": reason, "command": command.to_dict(),
            "candidates": candidates, "selected": selected,
            "selected_ids": [item["id"] for item in selected],
            # Keep the legacy single-object benchmark API.
            "selected_id": selected[0]["id"] if selected else None,
            "box": selected[0]["box"] if selected else None,
            "score": selected[0]["score"] if selected else None}


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def associate(track_boxes, detections, min_iou=0.3, affinities=None):
    """Maximum-weight one-to-one assignment with dummy unmatched columns.

    Hungarian assignment avoids depending on detector ordering or scipy. Lost
    tracks are deliberately excluded by the caller: re-entry receives a new ID.
    """
    ids = list(track_boxes)
    n, real_columns = len(ids), len(detections)
    if not n or not real_columns:
        return {}, list(range(real_columns))
    weights = affinities if affinities is not None else [[box_iou(track_boxes[obj_id], d["box"]) for d in detections] for obj_id in ids]
    m = real_columns + n
    costs = [[-w if w >= min_iou else 0.0 for w in row] + [0.0] * n for row in weights]
    u, v, p, way = [0.] * (n + 1), [0.] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minimum, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
    matches = {}
    for j in range(1, real_columns + 1):
        if p[j] and weights[p[j] - 1][j - 1] >= min_iou:
            matches[ids[p[j] - 1]] = j - 1
    return matches, [j for j in range(real_columns) if j not in matches.values()]
