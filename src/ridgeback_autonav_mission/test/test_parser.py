"""Unit tests for the mission task parser."""
from ridgeback_autonav_mission.lib.task_parser import normalize_room, parse_task


def test_parse_basic():
    assert parse_task('Go to Room 206') == '206'


def test_parse_with_letter_suffix():
    assert parse_task('find room 12B please') == '12B'


def test_parse_keyword_variants():
    assert parse_task('navigate to office #340') == '340'
    assert parse_task('head to suite 88') == '88'
    assert parse_task('go to the 1024 lab') == '1024'


def test_parse_none_when_no_number():
    assert parse_task('go to the kitchen') is None
    assert parse_task('') is None


def test_normalize_room():
    assert normalize_room(' 206 ') == '206'
    assert normalize_room('12b') == '12B'
    assert normalize_room('#3-40') == '340'
