"""Bounded-cost appearance and motion cues; not a learned ReID model."""
import cv2
import numpy as np

from .targets import box_iou


def appearance(rgb, box, mask=None):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    # Fixed 64x64 samples keep descriptor cost independent of source resolution.
    patch = cv2.resize(rgb[y1:y2, x1:x2], (64, 64))
    sample_mask = None
    if mask is not None:
        sample_mask = cv2.resize(mask[y1:y2, x1:x2].view(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], sample_mask, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return np.sqrt(hist / hist.sum()) if hist.sum() else None


def predicted_box(track, frame_index):
    delta = max(0, frame_index - track["motion_frame"])
    dx, dy = np.asarray(track["velocity"]) * delta
    x1, y1, x2, y2 = track["motion_box"]
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


def identity_affinity(track, candidate, descriptor, frame_index):
    predicted = predicted_box(track, frame_index)
    box = candidate["box"]
    a = np.array([(predicted[0] + predicted[2]) / 2, (predicted[1] + predicted[3]) / 2])
    b = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    scale = max(8, np.hypot(predicted[2] - predicted[0], predicted[3] - predicted[1]))
    distance = float(np.linalg.norm(a - b) / scale)
    area_a = max(1, (predicted[2] - predicted[0]) * (predicted[3] - predicted[1]))
    area_b = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    size_similarity = min(area_a, area_b) / max(area_a, area_b)
    stored = track.get("appearance")
    similarity = float(np.dot(stored, descriptor)) if stored is not None and descriptor is not None else 0.0
    overlap = max(box_iou(predicted, box), box_iou(track["box"], box) if track["state"] == "tracking" else 0)
    if stored is not None and descriptor is not None and similarity < .35:
        return 0.0
    if track["state"] == "occluded":
        # A short, spatially plausible recovery needs appearance evidence too.
        if similarity < .80 or distance > 1.0 or size_similarity < .2 or (box_iou(predicted, box) < .1 and distance > .35):
            return 0.0
    elif overlap < .1 and (similarity < .7 or distance > 1.25):
        return 0.0
    if size_similarity < .15:
        return 0.0
    return .45 * overlap + .35 * similarity + .20 * max(0, 1 - distance / 2)


def update_motion(track, box, frame_index, descriptor=None):
    old = track["motion_box"]
    delta = frame_index - track["motion_frame"]
    if delta > 0:
        velocity = [(box[0] + box[2] - old[0] - old[2]) / (2 * delta),
                    (box[1] + box[3] - old[1] - old[3]) / (2 * delta)]
        track["velocity"] = (np.asarray(track["velocity"]) * .4 + np.asarray(velocity) * .6).tolist()
    track["motion_box"], track["motion_frame"] = list(box), frame_index
    if descriptor is not None:
        previous = track.get("appearance")
        updated = descriptor if previous is None else previous * .9 + descriptor * .1
        norm = np.linalg.norm(updated)
        track["appearance"] = updated / norm if norm else descriptor
