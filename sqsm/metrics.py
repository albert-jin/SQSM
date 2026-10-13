"""Strict final-answer parsing; no gold-conditioned answer selection."""
from decimal import Decimal, InvalidOperation
import re


def extract_answer(text, task):
    if "####" in text:
        candidate = text.rsplit("####", 1)[1].strip()
    else:
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
        candidate = boxed[-1].strip() if boxed else text.strip().split("\n")[-1].strip()
    candidate = candidate.replace("**", "").strip()
    if task in {"aqua", "du"}:
        match = re.fullmatch(r"(?:[Tt]he (?:final )?answer is\s*)?\(?([A-Fa-f])\)?[.\s]*", candidate)
        return match.group(1).upper() if match else None
    if task == "strategyqa":
        match = re.fullmatch(r"(?:[Tt]he (?:final )?answer is\s*)?(yes|no)[.\s]*", candidate, re.I)
        return match.group(1).lower() if match else None
    candidate = candidate.replace(",", "").strip().rstrip(".")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+(?:\.\d+)?)?", candidate):
        return None
    try:
        if "/" in candidate:
            numerator, denominator = candidate.split("/")
            return str(Decimal(numerator) / Decimal(denominator))
        return str(Decimal(candidate).normalize())
    except (InvalidOperation, ZeroDivisionError):
        return None


def correct(prediction, gold, task):
    pred, expected = extract_answer(prediction, task), extract_answer("#### " + gold, task)
    if pred is None or expected is None:
        return False
    if task in {"gsm8k", "asdiv"}:
        return Decimal(pred) == Decimal(expected)
    return pred == expected
