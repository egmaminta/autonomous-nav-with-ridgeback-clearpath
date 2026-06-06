"""Parse a natural-language task into a target room number.

Regex-first (deterministic, no network). An optional LLM fallback can be wired
in for unusual phrasings, but it runs ONCE here — never inside the control loop
— so it cannot make the robot brittle. Pure python; unit-testable.
"""
import re

# Ordered: a number explicitly tied to a room/office/suite keyword wins over a
# bare number elsewhere in the sentence.
_PATTERNS = [
    re.compile(r'(?:room|rm|office|suite|unit|lab)\s*#?\s*([0-9]{1,4}[A-Za-z]?)', re.I),
    re.compile(r'\b([0-9]{2,4}[A-Za-z]?)\b'),
]


def normalize_room(token):
    return re.sub(r'[^0-9A-Za-z]', '', token).upper()


def parse_task(text):
    """Return the normalized target room string, or None if none is found."""
    if not text:
        return None
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            return normalize_room(m.group(1))
    return None


def parse_task_with_llm(text, model, anthropic_client=None):
    """Optional LLM fallback, used once and outside the control loop. ``model``
    is the model id; skipped unless ``anthropic_client`` is given."""
    room = parse_task(text)
    if room or anthropic_client is None:
        return room
    try:
        msg = anthropic_client.messages.create(
            model=model, max_tokens=20,
            messages=[{'role': 'user',
                       'content': f'Extract only the room/office number from this '
                                  f'instruction, no other text: "{text}"'}])
        return normalize_room(msg.content[0].text.strip())
    except Exception:  # noqa: BLE001
        return None
