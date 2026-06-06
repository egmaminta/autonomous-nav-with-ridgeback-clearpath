"""PARSeq scene-text recognizer wrapper.

Loaded from torch.hub (``baudm/parseq``) so no ``strhub`` pip install is needed;
the hub fetches the repo + pretrained weights to ``~/.cache/torch/hub`` on first
use (needs internet once). The preprocessing transform is rebuilt with
torchvision to avoid importing ``strhub`` directly. Torch is imported lazily so
this module imports without it.
"""


class ParseqReader:
    def __init__(self, hub='baudm/parseq', model='parseq', device='cuda:0',
                 min_conf=0.5, logger=None):
        import torch
        import torchvision.transforms as T
        self._torch = torch
        self._log = logger
        self.min_conf = min_conf
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load(hub, model, pretrained=True,
                                    trust_repo=True).eval().to(self.device)
        img_size = tuple(self.model.hparams.img_size)  # (H, W), e.g. (32, 128)
        self.transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

    def read(self, crop_bgr):
        """Recognize text in a BGR crop. Returns (text, confidence)."""
        from PIL import Image
        if crop_bgr.size == 0:
            return '', 0.0
        rgb = crop_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb.astype('uint8'), 'RGB')
        x = self.transform(pil).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            logits = self.model(x)
            probs = logits.softmax(-1)
            labels, confs = self.model.tokenizer.decode(probs)
        text = labels[0] if labels else ''
        conf = 0.0
        if confs:
            c = confs[0]
            try:
                import numpy as np
                arr = c.cpu().numpy() if hasattr(c, 'cpu') else np.asarray(c)
                # Geometric mean of per-char probabilities (robust to length,
                # unlike a raw product which underflows). Matches slam_burger.
                conf = float(np.exp(np.mean(np.log(np.clip(arr, 1e-6, 1.0))))) \
                    if arr.size else 0.0
            except Exception:  # noqa: BLE001
                conf = float(c)
        return text, conf
