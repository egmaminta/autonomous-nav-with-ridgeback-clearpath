"""Deduplicated, multi-observation sign registry (pure python, unit-testable).

Each detected room number is matched to an existing sign by (normalized text)
AND map-position within ``cluster_radius``. A sign is *confirmed* only after
``min_observations`` consistent hits whose positions stay within ``max_spread``
(a tight bound, << cluster_radius). Later observations EMA-refine the position.
The confirmation gate is what stops a single bad PARSeq read or one-off
misprojection from triggering a mission. A same-text detection farther than
``cluster_radius`` is treated as a *different* sign and must earn its own
confirmation, so a lone outlier never gets acted on.
"""
import math
import re


def normalize_text(text):
    """Uppercase, strip non-alphanumerics (so '206', ' 206 ', '#206' match)."""
    return re.sub(r'[^0-9A-Za-z]', '', text).upper()


class SignEntry:
    __slots__ = ('text', 'x', 'y', 'confidence', 'observations', 'confirmed',
                 '_samples', 'last_update')

    def __init__(self, text, x, y, conf, now):
        self.text = text
        self.x = x
        self.y = y
        self.confidence = conf
        self.observations = 1
        self.confirmed = False
        self._samples = [(x, y)]
        self.last_update = now


class SignRegistry:
    def __init__(self, cluster_radius=0.75, min_observations=3, max_spread=0.30,
                 position_ema=0.4):
        self.cluster_radius = cluster_radius
        self.min_observations = min_observations
        self.max_spread = max_spread
        self.ema = position_ema
        self.entries = []  # list[SignEntry]

    def observe(self, text, x, y, conf, now=0.0):
        """Fold in one observation. Returns the SignEntry it landed in."""
        key = normalize_text(text)
        if not key:
            return None
        match = None
        for e in self.entries:
            if e.text == key and math.hypot(e.x - x, e.y - y) <= self.cluster_radius:
                match = e
                break
        if match is None:
            e = SignEntry(key, x, y, conf, now)
            self.entries.append(e)
            return e

        match.observations += 1
        match.confidence = max(match.confidence, conf)
        match.x = (1 - self.ema) * match.x + self.ema * x
        match.y = (1 - self.ema) * match.y + self.ema * y
        match._samples.append((x, y))
        if len(match._samples) > 20:
            match._samples.pop(0)
        match.last_update = now
        if (not match.confirmed and match.observations >= self.min_observations
                and self._spread(match) <= self.max_spread):
            match.confirmed = True
        return match

    def _spread(self, e):
        xs = [s[0] for s in e._samples]
        ys = [s[1] for s in e._samples]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def find(self, text, confirmed_only=True):
        """Return the SignEntry matching ``text`` (or None)."""
        key = normalize_text(text)
        for e in self.entries:
            if e.text == key and (e.confirmed or not confirmed_only):
                return e
        return None

    def all(self, confirmed_only=False):
        return [e for e in self.entries if e.confirmed or not confirmed_only]
