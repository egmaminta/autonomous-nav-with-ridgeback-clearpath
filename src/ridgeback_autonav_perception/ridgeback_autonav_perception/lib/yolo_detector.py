"""YOLO sign / text-region detector wrapper (ultralytics).

Default weights are the same text-region checkpoint the old slam_burger v2 stack
used: ``armvectores/yolov8n_handwritten_text_detection`` (file ``best.pt``), a
YOLOv8-nano fine-tuned for text detection. It is fetched from the HuggingFace
Hub on first use and cached on disk. For best room-sign accuracy, fine-tune on
your own door-plate crops and point ``weights`` at the local ``.pt``.

Torch + ultralytics import lazily inside ``__init__`` so this module imports in
environments without them (e.g. pure-math unit tests).
"""
import os


def filter_and_merge_boxes(boxes, img_w, img_h, merge_gap=20.0,
                           min_area=200.0, max_area_frac=0.4):
    """Merge nearby/overlapping detection boxes into one region per sign, then
    drop specks and oversized background regions.

    The text detector tends to fragment a printed sign ("RM 205") into slices
    and fire on clutter. Unioning boxes whose (gap-expanded) rectangles touch
    gives PARSeq the whole sign in one crop; the area filter removes tiny noise
    and whole-wall regions. Returns 6-tuples (x1, y1, x2, y2, conf, cls).
    """
    rects = [[b[0], b[1], b[2], b[3], b[4]] for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            x1, y1, x2, y2, c = rects[i]
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                a1, b1, a2, b2, cj = rects[j]
                # do the gap-expanded rectangles overlap?
                if not (x1 - merge_gap > a2 or a1 - merge_gap > x2 or
                        y1 - merge_gap > b2 or b1 - merge_gap > y2):
                    x1, y1 = min(x1, a1), min(y1, b1)
                    x2, y2 = max(x2, a2), max(y2, b2)
                    c = max(c, cj)
                    used[j] = True
                    changed = True
            used[i] = True
            out.append([x1, y1, x2, y2, c])
        rects = out
    max_area = max_area_frac * img_w * img_h
    res = []
    for x1, y1, x2, y2, c in rects:
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        if min_area <= area <= max_area:
            res.append((x1, y1, x2, y2, c, 0))
    return res


DEFAULT_HF_REPO = 'armvectores/yolov8n_handwritten_text_detection'
DEFAULT_HF_FILENAME = 'best.pt'
_MODEL_EXTS = ('.pt', '.pth', '.onnx', '.engine')
# First dir is ours; second reuses a checkpoint already pulled by slam_burger.
_CACHE_DIRS = (
    os.path.expanduser('~/.cache/ridgeback_autonav/yolo'),
    os.path.expanduser('~/.cache/slam_burger/text_landmark_v2'),
)


def resolve_weights(weights, hf_filename='', logger=None):
    """Resolve a weights spec to a usable path / ultralytics handle.

    Order: existing local file or ultralytics shorthand (``*.pt``) -> env
    ``RIDGEBACK_AUTONAV_YOLO_WEIGHTS`` -> HuggingFace repo id (the spec itself, or the
    default text-detection repo when empty), cached on disk.
    """
    w = (weights or '').strip()
    if w.endswith(_MODEL_EXTS):
        # Local path, or a bare ultralytics name like 'yolov8n.pt' (auto-download).
        return w
    env = os.environ.get('RIDGEBACK_AUTONAV_YOLO_WEIGHTS', '').strip()
    if env and os.path.exists(env):
        return env

    repo = w or DEFAULT_HF_REPO
    fname = hf_filename or DEFAULT_HF_FILENAME
    safe = repo.replace('/', '__') + '__' + fname
    for cdir in _CACHE_DIRS:
        cached = os.path.join(cdir, safe)
        if os.path.exists(cached):
            if logger:
                logger.info(f'using cached YOLO weights: {cached}')
            return cached

    from huggingface_hub import hf_hub_download
    dest_dir = _CACHE_DIRS[0]
    os.makedirs(dest_dir, exist_ok=True)
    if logger:
        logger.info(f'downloading YOLO weights {repo}/{fname} (first use)...')
    downloaded = hf_hub_download(repo_id=repo, filename=fname, cache_dir=dest_dir)
    dest = os.path.join(dest_dir, safe)
    try:
        import shutil
        if os.path.abspath(downloaded) != os.path.abspath(dest):
            shutil.copyfile(downloaded, dest)
        return dest
    except Exception:  # noqa: BLE001
        return downloaded


class YoloDetector:
    def __init__(self, weights='', conf=0.35, iou=0.45, imgsz=640,
                 device='cuda:0', hf_filename='', logger=None,
                 merge=True, merge_gap=20.0, min_area=200.0, max_area_frac=0.4):
        import torch
        from ultralytics import YOLO
        self._log = logger
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.merge = merge
        self.merge_gap = merge_gap
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        path = resolve_weights(weights, hf_filename, logger)
        self.model = YOLO(path)
        if logger:
            logger.info(f'YOLO detector ready: {os.path.basename(str(path))} '
                        f'on {self.device}')

    def detect(self, image_bgr):
        """Return a list of (x1, y1, x2, y2, conf, cls) boxes in pixels."""
        results = self.model.predict(
            image_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=self.device, verbose=False)
        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append((float(x1), float(y1), float(x2), float(y2),
                              float(b.conf[0]), int(b.cls[0])))
        if self.merge and boxes:
            h, w = image_bgr.shape[:2]
            boxes = filter_and_merge_boxes(boxes, w, h, self.merge_gap,
                                           self.min_area, self.max_area_frac)
        return boxes
