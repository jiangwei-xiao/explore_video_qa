import math
import re

PROTOCOL_VERSION = 'videoqa-llava-16-v1'


def options_text(options):
    if len(options) != 4:
        raise ValueError('This protocol requires four options')
    formatted = []
    for letter, option in zip('ABCD', options):
        text = option.strip()
        match = re.match(r'^([A-D])[.)]\s*', text)
        if match:
            if match.group(1) != letter:
                raise ValueError('Option prefixes disagree with their original order')
            text = text[match.end():]
        formatted.append(f'{letter}. {text}')
    return '\n'.join(formatted)


def question_text(question, options, duration, times):
    if len(times) != 16 or len(set(times)) != 16 or list(times) != sorted(times):
        raise ValueError('Expected 16 unique timestamps in chronological order')
    if not math.isfinite(duration) or duration <= 0 or any(not math.isfinite(t) or t < 0 or t >= duration for t in times):
        raise ValueError('Invalid duration or source timestamp')
    content = question + '\n' + '\n'.join(options)
    if any(token in content for token in ('<image>', '<im_start>', '<im_end>')):
        raise ValueError('Unexpected control token in dataset text')
    locations = ', '.join(f'{t:.2f}s' for t in times)
    return (f'<image>\nThe video lasts for {duration:.2f} seconds, and 16 frames are sampled from it. '
            f'These frames are located at {locations}. Please answer the following question related to this video.\n'
            f'{question}\n{options_text(options)}\nAnswer with only the option letter (A, B, C, or D).')


def parse_answer(text):
    match = re.fullmatch(r'(?:Answer\s*:\s*)?(?:\(([A-D])\)|([A-D]))[.]?', text.strip(), re.IGNORECASE)
    return (match.group(1) or match.group(2)).upper() if match else None
