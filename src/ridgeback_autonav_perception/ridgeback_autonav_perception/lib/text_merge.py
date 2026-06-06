"""Group word-level text detections into phrase-level ones (pure, unit-testable).

Ported from slam_burger's text_detection_v2. A printed sign like "RM 205" is
detected as separate word boxes; recognizing each tight box with PARSeq and then
merging the recognized *texts* by line + adjacency yields a stable, readable
phrase — far more reliable than running the recognizer on one messy merged crop.

Items are dicts: {'text': str, 'confidence': float, 'bbox_rect': (x1,y1,x2,y2)}.
"""


def _line_overlap_ratio(rect_a, rect_b):
    _, ay1, _, ay2 = rect_a
    _, by1, _, by2 = rect_b
    overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
    ah = max(1.0, ay2 - ay1)
    bh = max(1.0, by2 - by1)
    return float(overlap / min(ah, bh))


def merge_text_detections(items):
    """Merge word-level detections into phrase-level detections by text line."""
    if len(items) <= 1:
        return list(items)

    lines = []
    by_y = sorted(items, key=lambda d: ((d['bbox_rect'][1] + d['bbox_rect'][3]) * 0.5,
                                        d['bbox_rect'][0]))
    for item in by_y:
        x1, y1, x2, y2 = item['bbox_rect']
        yc = (y1 + y2) * 0.5
        h = max(1.0, y2 - y1)
        matched = None
        for line in lines:
            lrect = (line['x_min'], line['y_min'], line['x_max'], line['y_max'])
            if _line_overlap_ratio(item['bbox_rect'], lrect) < 0.30:
                continue
            lh = max(1.0, line['y_max'] - line['y_min'])
            if abs(yc - line['y_center']) > 0.65 * max(h, lh):
                continue
            matched = line
            break
        if matched is None:
            lines.append({'items': [item], 'x_min': x1, 'x_max': x2,
                          'y_min': y1, 'y_max': y2, 'y_center': yc})
            continue
        matched['items'].append(item)
        matched['x_min'] = min(matched['x_min'], x1)
        matched['x_max'] = max(matched['x_max'], x2)
        matched['y_min'] = min(matched['y_min'], y1)
        matched['y_max'] = max(matched['y_max'], y2)
        matched['y_center'] = (matched['y_min'] + matched['y_max']) * 0.5

    merged = []
    for line in lines:
        tokens = sorted(line['items'], key=lambda d: d['bbox_rect'][0])
        if not tokens:
            continue
        current = dict(tokens[0])
        for nxt in tokens[1:]:
            cx1, cy1, cx2, cy2 = current['bbox_rect']
            nx1, ny1, nx2, ny2 = nxt['bbox_rect']
            h_ref = max(1.0, cy2 - cy1, ny2 - ny1)
            gap = nx1 - cx2
            if gap <= max(10.0, 1.10 * h_ref):
                sep = ' ' if gap >= 0.20 * h_ref else ''
                current['text'] = (str(current.get('text', '')).strip() + sep +
                                   str(nxt.get('text', '')).strip()).strip()
                c_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))
                n_area = max(1.0, (nx2 - nx1) * (ny2 - ny1))
                current['confidence'] = float(
                    (current['confidence'] * c_area + nxt['confidence'] * n_area)
                    / (c_area + n_area))
                current['bbox_rect'] = (min(cx1, nx1), min(cy1, ny1),
                                        max(cx2, nx2), max(cy2, ny2))
            else:
                merged.append(current)
                current = dict(nxt)
        merged.append(current)

    merged = [m for m in merged if str(m.get('text', '')).strip()]
    merged.sort(key=lambda d: ((d['bbox_rect'][1] + d['bbox_rect'][3]) * 0.5,
                               d['bbox_rect'][0]))
    return merged
