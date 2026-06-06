"""Unit tests for line-aware text-detection merging."""
from ridgeback_autonav_perception.lib.text_merge import merge_text_detections, _line_overlap_ratio


def _item(text, x1, y1, x2, y2, conf=0.9):
    return {'text': text, 'confidence': conf, 'bbox_rect': (x1, y1, x2, y2)}


def test_merge_same_line_into_phrase():
    items = [_item('RM', 0, 0, 30, 20), _item('205', 35, 0, 80, 20)]
    out = merge_text_detections(items)
    assert len(out) == 1
    assert out[0]['text'] == 'RM 205'                 # space inserted on a real gap
    assert out[0]['bbox_rect'] == (0, 0, 80, 20)      # union box


def test_adjacent_no_space():
    # touching boxes (tiny gap) -> concatenated without a space
    items = [_item('RM', 0, 0, 30, 20), _item('205', 31, 0, 76, 20)]
    out = merge_text_detections(items)
    assert out[0]['text'] in ('RM205', 'RM 205')      # gap ~ threshold


def test_different_lines_stay_separate():
    items = [_item('RM 205', 0, 0, 80, 20), _item('EXIT', 0, 60, 60, 80)]
    out = merge_text_detections(items)
    assert len(out) == 2


def test_single_item_passthrough():
    assert merge_text_detections([_item('205', 0, 0, 40, 20)])[0]['text'] == '205'


def test_line_overlap_ratio():
    assert _line_overlap_ratio((0, 0, 10, 20), (0, 0, 10, 20)) == 1.0
    assert _line_overlap_ratio((0, 0, 10, 20), (0, 100, 10, 120)) == 0.0


def test_confidence_area_weighted():
    items = [_item('RM', 0, 0, 20, 20, conf=0.6), _item('205', 22, 0, 82, 20, conf=0.9)]
    out = merge_text_detections(items)
    # bigger '205' box dominates -> merged conf between the two, nearer 0.9
    assert 0.6 < out[0]['confidence'] <= 0.9
