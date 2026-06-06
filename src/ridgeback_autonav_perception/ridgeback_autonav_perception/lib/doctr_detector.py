"""DocTR (DBNet) text-region detector — a proper printed/scene-text detector.

Replaces the handwritten-text YOLO that fragmented printed room signs and
false-positived on clutter. Returns clean per-word boxes which are recognized
individually by PARSeq and merged into phrases. Torch + doctr import lazily.
Mirrors slam_burger's DocTRDetector.
"""


class DocTRDetector:
    def __init__(self, arch='db_mobilenet_v3_large', device='cuda:0',
                 input_size=512, logger=None):
        import torch
        from doctr.models import detection_predictor
        self._log = logger
        dev = 'cuda' if (device.startswith('cuda') and torch.cuda.is_available()) else 'cpu'
        self.model = detection_predictor(arch=arch, pretrained=True,
                                         assume_straight_pages=True)
        try:
            self.model = self.model.to(dev)
        except Exception:  # noqa: BLE001
            dev = 'cpu'
        # DocTR defaults to a 1024 resize; 512 halves inference time on Jetson
        # with no measurable recall loss for large signage.
        try:
            if input_size:
                resize = getattr(self.model.pre_processor, 'resize', None)
                if resize is not None and hasattr(resize, 'size'):
                    resize.size = (int(input_size), int(input_size))
        except Exception:  # noqa: BLE001
            pass
        self.device = dev
        if logger:
            logger.info(f'DocTR detector ready: {arch} on {dev} (input={input_size})')

    def detect(self, image_bgr):
        """Return per-word boxes as (x1, y1, x2, y2, score) in pixels."""
        import numpy as np
        rgb = image_bgr[:, :, ::-1]
        h, w = rgb.shape[:2]
        out = self.model([np.ascontiguousarray(rgb)])
        if not out:
            return []
        page = out[0]
        words = page.get('words') if isinstance(page, dict) else page
        if words is None:
            return []
        words = np.asarray(words, dtype=np.float32)
        if words.size == 0:
            return []
        boxes = []
        for row in words:                      # coords normalised in [0,1]
            x1, y1, x2, y2 = row[:4]
            score = float(row[4]) if row.shape[0] > 4 else 1.0
            boxes.append((float(x1 * w), float(y1 * h), float(x2 * w),
                          float(y2 * h), score, 0))
        return boxes
