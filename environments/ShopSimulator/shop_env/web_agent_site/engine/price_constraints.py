"""Deterministic Chinese price-constraint compilation for Reward v4."""

from __future__ import annotations

import math
import re


PRICE_CONSTRAINT_VERSION = "shopping-price-constraint-v2"
_PRICE_PREFIX = r"(?:预算|价格|售价|价钱|价位|总价|费用|成本|花费)"
_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万亿]+)"
_UNIT = r"(?:块钱|人民币|元|块|万|千|[kK])?"
_MONEY = rf"(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})"
_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_LARGE_UNITS = {"万": 10000, "亿": 100000000}


def _chinese_integer(value: str) -> int | None:
    if not value or any(
        character not in _CHINESE_DIGITS | _SMALL_UNITS | _LARGE_UNITS
        for character in value
    ):
        return None
    total = 0
    section = 0
    digit = 0
    for character in value:
        if character in _CHINESE_DIGITS:
            digit = _CHINESE_DIGITS[character]
        elif character in _SMALL_UNITS:
            unit = _SMALL_UNITS[character]
            section += (digit or 1) * unit
            digit = 0
        else:
            section += digit
            total = (total + section) * _LARGE_UNITS[character]
            section = 0
            digit = 0
    amount = total + section + digit
    # Colloquial prices omit the final smaller unit: 五万七=57000,
    # 一万二千三=12300. An explicit 零 keeps the conventional literal parse.
    if value[-1] in _CHINESE_DIGITS and len(value) > 1:
        unit_positions = [
            (index, _SMALL_UNITS.get(character) or _LARGE_UNITS.get(character))
            for index, character in enumerate(value[:-1])
            if character in _SMALL_UNITS or character in _LARGE_UNITS
        ]
        if unit_positions:
            last_index, last_unit = unit_positions[-1]
            suffix = value[last_index + 1 :]
            if (
                len(suffix) == 1
                and "零" not in value[last_index + 1 :]
                and "〇" not in value[last_index + 1 :]
                and last_unit >= 100
            ):
                trailing = _CHINESE_DIGITS[suffix]
                amount += trailing * (last_unit // 10) - trailing
    return amount


def parse_money_number(value: object, unit: object = "") -> float | None:
    """Parse an Arabic or conventional Chinese money amount."""

    raw = str(value or "").strip().replace(",", "")
    normalized_unit = str(unit or "").strip().casefold()
    if not raw:
        return None
    try:
        amount = float(raw)
    except ValueError:
        integer = _chinese_integer(raw)
        amount = float(integer) if integer is not None else math.nan
    if normalized_unit == "万":
        amount *= 10000
    elif normalized_unit in {"千", "k"}:
        amount *= 1000
    return amount if math.isfinite(amount) and amount > 0 else None


def _money(match: re.Match, prefix: str = "") -> float | None:
    return parse_money_number(
        match.group(f"{prefix}value"), match.group(f"{prefix}unit")
    )


def _constraint(
    *,
    kind: str,
    source: str,
    lower: float | None = None,
    upper: float | None = None,
    target: float | None = None,
    tolerance: float | None = None,
) -> dict:
    return {
        "version": PRICE_CONSTRAINT_VERSION,
        "kind": kind,
        "lower": lower,
        "upper": upper,
        "target": target,
        "tolerance": tolerance,
        "source_text": source,
    }


def compile_price_constraint(instruction: object) -> dict | None:
    """Compile explicit price language without using the target product price."""

    raw = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(instruction or ""))
    raw = re.sub(r"\s+", "", raw)
    if not raw:
        return None

    clauses = [
        clause
        for clause in re.split(r"[，。；;,!?！？]|(?<!\d)\.(?!\d)", raw)
        if clause
    ]
    price_clauses = []
    for clause in clauses:
        prefix = re.search(_PRICE_PREFIX, clause)
        if prefix:
            # Ignore earlier numeric size/count/capacity requirements in the
            # same clause; price syntax starts at the explicit price marker.
            first_number = re.search(_NUMBER, clause)
            price_clauses.append(
                clause[prefix.start() :]
                if first_number is None or prefix.start() < first_number.start()
                else clause
            )
        elif re.search(r"(?:元|块钱|人民币|块)(?:左右|上下|以内|以下|以上)?", clause):
            price_clauses.append(clause)
        elif re.search(r"[零〇一二两三四五六七八九十百千万亿]+以内(?:能)?搞定", clause):
            price_clauses.append(clause)
    for text in price_clauses:
        result = _compile_price_clause(text)
        if result is not None:
            return result
    return None


def _compile_price_clause(text: str) -> dict | None:
    """Compile one clause already identified as describing price."""

    range_pattern = re.compile(
        rf"{_PRICE_PREFIX}(?:控制)?(?:在)?"
        rf"(?P<low_value>{_NUMBER})(?P<low_unit>{_UNIT})"
        rf"(?:-|~|～|至|到)"
        rf"(?P<high_value>{_NUMBER})(?P<high_unit>{_UNIT})(?:之间|范围内)?"
        rf"(?![\d.])"
    )
    match = range_pattern.search(text)
    if match:
        low_unit = match.group("low_unit")
        high_unit = match.group("high_unit")
        if not low_unit and high_unit in {"万", "千", "k", "K"}:
            low_unit = high_unit
        if not high_unit and low_unit in {"万", "千", "k", "K"}:
            high_unit = low_unit
        low = parse_money_number(match.group("low_value"), low_unit)
        high = parse_money_number(match.group("high_value"), high_unit)
        if low is not None and high is not None and low <= high:
            return _constraint(
                kind="hard_range",
                source=match.group(0),
                lower=low,
                upper=high,
            )

    maximum_patterns = (
        re.compile(rf"(?:不超过|不高于|最高|最多){_MONEY}"),
        re.compile(rf"{_MONEY}(?:以内|以下|内)"),
    )
    for pattern in maximum_patterns:
        match = pattern.search(text)
        if match:
            value = _money(match)
            if value is not None:
                return _constraint(
                    kind="hard_max", source=match.group(0), upper=value
                )

    minimum_patterns = (
        re.compile(rf"(?:不少于|不低于|至少|最低){_MONEY}"),
        re.compile(rf"{_MONEY}(?:以上|起|\+)"),
    )
    for pattern in minimum_patterns:
        match = pattern.search(text)
        if match:
            value = _money(match)
            if value is not None:
                return _constraint(
                    kind="hard_min", source=match.group(0), lower=value
                )

    soft_patterns = (
        re.compile(
            rf"{_PRICE_PREFIX}(?:控制)?(?:在)?(?:大约|大概|约)?"
            rf"{_MONEY}(?:左右|上下|附近|出头|多点|多)"
        ),
        re.compile(rf"{_PRICE_PREFIX}(?:大约|大概|约){_MONEY}"),
        re.compile(rf"(?:大约|大概|约){_MONEY}(?:左右|上下|附近)?"),
    )
    for pattern in soft_patterns:
        match = pattern.search(text)
        if match:
            value = _money(match)
            if value is not None:
                tolerance = max(5.0, value * 0.10)
                return _constraint(
                    kind="soft_target",
                    source=match.group(0),
                    lower=max(0.0, value - tolerance),
                    upper=value + tolerance,
                    target=value,
                    tolerance=tolerance,
                )

    plain = re.search(rf"{_PRICE_PREFIX}(?:控制)?(?:在)?{_MONEY}", text)
    if plain:
        value = _money(plain)
        if value is not None:
            return _constraint(
                kind="hard_max", source=plain.group(0), upper=value
            )
    colloquial_max = re.search(
        rf"(?P<value>[零〇一二两三四五六七八九十百千万亿]+)"
        rf"(?P<unit>)(?:以内)(?:能)?搞定",
        text,
    )
    if colloquial_max:
        value = _money(colloquial_max)
        if value is not None:
            return _constraint(
                kind="hard_max",
                source=colloquial_max.group(0),
                upper=value,
            )
    return None
