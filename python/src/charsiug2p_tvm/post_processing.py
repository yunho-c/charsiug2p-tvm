from __future__ import annotations

SUPPORTED_STRATEGIES = {
    "espeak",
    "ipa",
    "ipa-flap",
    "ipa-vowel",
    "ipa-flap-vowel",
    "ipa-vowel-stress",
    "ipa-flap-vowel-stress",
    "ipa-vowel-syllabic",
    "ipa-flap-vowel-syllabic",
    "ipa-vowel-reduced",
    "ipa-flap-vowel-reduced",
    "ipa-vowel-prefix",
    "ipa-flap-vowel-prefix",
}

_FROM_ESPEAKS = sorted(
    {
        "\u0303": "",
        "a^ɪ": "I",
        "a^ʊ": "W",
        "d^ʒ": "ʤ",
        "e": "A",
        "e^ɪ": "A",
        "r": "ɹ",
        "t^ʃ": "ʧ",
        "x": "k",
        "ç": "k",
        "ɐ": "ə",
        "ɔ^ɪ": "Y",
        "ə^l": "ᵊl",
        "ɚ": "əɹ",
        "ɬ": "l",
        "ʔ": "t",
        "ʔn": "tᵊn",
        "ʔˌn\u0329": "tᵊn",
        "ʲ": "",
        "ʲO": "jO",
        "ʲQ": "jQ",
    }.items(),
    key=lambda kv: -len(kv[0]),
)

_VOWELS = {
    "ɑ",
    "ɔ",
    "ɛ",
    "ɜ",
    "ɪ",
    "ʊ",
    "ʌ",
    "ə",
    "i",
    "u",
    "A",
    "I",
    "W",
    "Y",
    "O",
    "Q",
    "æ",
    "a",
    "ɒ",
    "ᵻ",
    "ᵊ",
}

_IPA_REWRITES_BASE = sorted(
    {
        "tʃ": "t^ʃ",
        "dʒ": "d^ʒ",
        "oʊ": "o^ʊ",
        "əʊ": "ə^ʊ",
        "aɪ": "a^ɪ",
        "aʊ": "a^ʊ",
        "eɪ": "e^ɪ",
        "ɔɪ": "ɔ^ɪ",
        "ɝ": "ɜɹ",
        "ɫ": "l",
        "əl": "ə^l",
        "ən": "ᵊn",
        "əm": "ᵊm",
    }.items(),
    key=lambda kv: -len(kv[0]),
)

_IPA_REWRITES_VOWEL = sorted(
    {
        "tʃ": "t^ʃ",
        "dʒ": "d^ʒ",
        "oʊ": "o^ʊ",
        "əʊ": "ə^ʊ",
        "aɪ": "a^ɪ",
        "aʊ": "a^ʊ",
        "eɪ": "e^ɪ",
        "ɔɪ": "ɔ^ɪ",
        "ɝ": "ɜɹ",
        "ɫ": "l",
        "əl": "ə^l",
        "əɫ": "ə^l",
    }.items(),
    key=lambda kv: -len(kv[0]),
)

_IPA_REWRITES_VOWEL_SYLLABIC = sorted(
    {
        "tʃ": "t^ʃ",
        "dʒ": "d^ʒ",
        "oʊ": "o^ʊ",
        "əʊ": "ə^ʊ",
        "aɪ": "a^ɪ",
        "aʊ": "a^ʊ",
        "eɪ": "e^ɪ",
        "ɔɪ": "ɔ^ɪ",
        "ɝ": "ɜɹ",
        "ɫ": "l",
    }.items(),
    key=lambda kv: -len(kv[0]),
)

_IPA_STRATEGIES = {
    "ipa",
    "ipa-flap",
    "ipa-vowel",
    "ipa-flap-vowel",
    "ipa-vowel-stress",
    "ipa-flap-vowel-stress",
    "ipa-vowel-syllabic",
    "ipa-flap-vowel-syllabic",
    "ipa-vowel-reduced",
    "ipa-flap-vowel-reduced",
    "ipa-vowel-prefix",
    "ipa-flap-vowel-prefix",
}
_VOWEL_TUNED_STRATEGIES = {
    "ipa-vowel",
    "ipa-flap-vowel",
    "ipa-vowel-stress",
    "ipa-flap-vowel-stress",
    "ipa-vowel-syllabic",
    "ipa-flap-vowel-syllabic",
    "ipa-vowel-reduced",
    "ipa-flap-vowel-reduced",
    "ipa-vowel-prefix",
    "ipa-flap-vowel-prefix",
}
_STRESS_TUNED_STRATEGIES = {"ipa-vowel-stress", "ipa-flap-vowel-stress"}
_SYLLABIC_TUNED_STRATEGIES = {"ipa-vowel-syllabic", "ipa-flap-vowel-syllabic"}
_REDUCED_TUNED_STRATEGIES = {"ipa-vowel-reduced", "ipa-flap-vowel-reduced"}
_PREFIX_STRESS_STRATEGIES = {"ipa-vowel-prefix", "ipa-flap-vowel-prefix"}
_FLAP_STRATEGIES = {
    "ipa-flap",
    "ipa-flap-vowel",
    "ipa-flap-vowel-stress",
    "ipa-flap-vowel-syllabic",
    "ipa-flap-vowel-reduced",
}
_STRESS_MARKERS = {"ˈ", "ˌ"}
_STRESS_TOKEN_UNITS = ["ᵊl", "ᵊn", "ᵊm", "ɜɹ", "əɹ"]
_REDUCED_STRESS_VOWELS = {"ɪ", "ᵻ", "ə", "ᵊ"}
_VOWEL_TOKENS = _VOWELS | {"ɜɹ", "əɹ", "ᵊl", "ᵊn", "ᵊm"}
STRESS_PREFIXES = (
    "under",
    "over",
    "inter",
    "intra",
    "intro",
    "super",
    "sub",
    "anti",
    "auto",
    "counter",
    "trans",
    "tele",
    "pre",
    "pro",
    "con",
    "com",
    "dis",
    "mis",
    "non",
    "out",
    "per",
    "re",
    "un",
    "in",
    "im",
    "il",
    "ir",
)
_STRESS_DROP_PREFIXES = {"re", "un", "in", "im", "il", "ir"}


def normalize_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower().replace("_", "-").replace("+", "-")
    if normalized == "espeak":
        return normalized
    parts = [part for part in normalized.split("-") if part]
    if "ipa" in parts:
        has_flap = "flap" in parts or "flaps" in parts
        has_vowel = "vowel" in parts or "vowels" in parts
        has_stress = "stress" in parts or "stresses" in parts
        has_prefix = "prefix" in parts or "pref" in parts
        has_syllabic = "syllabic" in parts or "schwa" in parts
        has_reduced = "reduced" in parts or "reduce" in parts
        if has_prefix:
            has_vowel = True
        if has_syllabic:
            has_vowel = True
        if has_reduced:
            has_vowel = True
        if has_flap and has_vowel and has_stress:
            normalized = "ipa-flap-vowel-stress"
        elif has_vowel and has_stress:
            normalized = "ipa-vowel-stress"
        elif has_flap and has_syllabic:
            normalized = "ipa-flap-vowel-syllabic"
        elif has_syllabic:
            normalized = "ipa-vowel-syllabic"
        elif has_flap and has_reduced:
            normalized = "ipa-flap-vowel-reduced"
        elif has_reduced:
            normalized = "ipa-vowel-reduced"
        elif has_flap and has_prefix:
            normalized = "ipa-flap-vowel-prefix"
        elif has_prefix:
            normalized = "ipa-vowel-prefix"
        elif has_flap and has_vowel:
            normalized = "ipa-flap-vowel"
        elif has_vowel:
            normalized = "ipa-vowel"
        elif has_flap:
            normalized = "ipa-flap"
        else:
            normalized = "ipa"
    if normalized not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return normalized


def _swap_syllabic_markers(text: str) -> str:
    chars = list(text)
    i = 0
    while i < len(chars):
        if i + 1 < len(chars) and chars[i + 1] == "\u0329":
            consonant = chars[i]
            chars[i] = "ᵊ"
            chars[i + 1] = consonant
            i += 2
        else:
            i += 1
    return "".join(chars).replace("\u0329", "")


def _apply_ipa_rewrites(text: str, rewrites: list[tuple[str, str]]) -> str:
    text = text.replace("\u0361", "^")
    for old, new in rewrites:
        text = text.replace(old, new)
    return text


def _apply_flap(text: str) -> str:
    chars = list(text)
    result: list[str] = []
    i = 0
    last_vowel = False
    while i < len(chars):
        char = chars[i]
        if char in {"ˈ", "ˌ"}:
            result.append(char)
            i += 1
            continue
        if char == "t" and last_vowel:
            j = i + 1
            while j < len(chars) and chars[j] in {"ˈ", "ˌ"}:
                j += 1
            if j < len(chars) and chars[j] in _VOWELS:
                result.append("ɾ")
                i += 1
                last_vowel = False
                continue
        last_vowel = char in _VOWELS
        result.append(char)
        i += 1
    return "".join(result)


def _tokenize_for_stress(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(text):
        if text[i] in _STRESS_MARKERS:
            tokens.append(text[i])
            i += 1
            continue
        matched = False
        for unit in _STRESS_TOKEN_UNITS:
            if text.startswith(unit, i):
                tokens.append(unit)
                i += len(unit)
                matched = True
                break
        if matched:
            continue
        tokens.append(text[i])
        i += 1
    return tokens


def _next_vowel_token(tokens: list[str], start: int) -> str | None:
    for idx in range(start, len(tokens)):
        if tokens[idx] in _VOWEL_TOKENS:
            return tokens[idx]
    return None


def _apply_stress_normalization(text: str) -> str:
    tokens = _tokenize_for_stress(text)
    if not tokens:
        return text

    has_primary = "ˈ" in tokens
    saw_primary = False
    output: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _STRESS_MARKERS:
            next_vowel = _next_vowel_token(tokens, i + 1)
            if token == "ˈ":
                if saw_primary:
                    if next_vowel in _REDUCED_STRESS_VOWELS:
                        i += 1
                        continue
                    output.append("ˌ")
                else:
                    output.append("ˈ")
                    saw_primary = True
                i += 1
                continue
            if has_primary and next_vowel in _REDUCED_STRESS_VOWELS:
                i += 1
                continue
            output.append("ˌ")
            i += 1
            continue
        output.append(token)
        i += 1
    return "".join(output)


def _match_stress_prefix(word: str | None) -> str | None:
    if not word:
        return None
    lowered = word.lower()
    for prefix in STRESS_PREFIXES:
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            return prefix
    return None


def _stress_signature(text: str) -> list[str]:
    return [char for char in text if char in _STRESS_MARKERS]


def _first_stress(text: str) -> str | None:
    for char in text:
        if char in _STRESS_MARKERS:
            return char
    return None


def _apply_prefix_stress_rules(text: str, word: str | None) -> str:
    prefix = _match_stress_prefix(word)
    if prefix is None:
        return text
    signature = _stress_signature(text)
    if signature.count("ˈ") == 1 and signature.count("ˌ") == 1:
        primary_idx = text.find("ˈ")
        secondary_idx = text.find("ˌ")
        if 0 <= primary_idx < secondary_idx:
            chars = list(text)
            chars[primary_idx] = "ˌ"
            chars[secondary_idx] = "ˈ"
            text = "".join(chars)
    if prefix in _STRESS_DROP_PREFIXES and signature.count("ˈ") == 1 and signature.count("ˌ") == 1:
        first_stress = _first_stress(text)
        if first_stress == "ˌ":
            text = text.replace("ˌ", "", 1)
    return text


def _apply_vowel_tuning(text: str) -> str:
    result: list[str] = []
    pending_stress: str | None = None
    i = 0
    while i < len(text):
        char = text[i]
        if char in {"ˈ", "ˌ"}:
            if pending_stress:
                result.append(pending_stress)
            pending_stress = char
            i += 1
            continue

        if text.startswith("ɜɹ", i) or text.startswith("əɹ", i):
            stressed = pending_stress is not None
            if pending_stress:
                result.append(pending_stress)
                pending_stress = None
            result.append("ɜɹ" if stressed else "əɹ")
            i += 2
            continue

        if text.startswith("ɪd", i) or text.startswith("ɪz", i):
            if pending_stress:
                result.append(pending_stress)
                pending_stress = None
                result.append(text[i : i + 2])
            else:
                result.append(f"ᵻ{text[i + 1]}")
            i += 2
            continue

        if char == "ə":
            if pending_stress:
                result.append(pending_stress)
                pending_stress = None
                result.append("ʌ")
            else:
                result.append(char)
            i += 1
            continue

        if char == "ᵊ":
            if pending_stress:
                result.append(pending_stress)
                pending_stress = None
                result.append("ə")
            else:
                result.append(char)
            i += 1
            continue

        if pending_stress and char in _VOWELS:
            result.append(pending_stress)
            pending_stress = None
        result.append(char)
        i += 1

    if pending_stress:
        result.append(pending_stress)
    return "".join(result)


def _apply_syllabic_tuning(text: str) -> str:
    result: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "ə":
            next_char = text[i + 1] if i + 1 < len(text) else ""
            if next_char in {"l", "n"}:
                prev_char = text[i - 1] if i > 0 else ""
                next_next = text[i + 2] if i + 2 < len(text) else ""
                prev_is_vowel = prev_char in _VOWELS
                next_is_vowel = next_next in _VOWELS or next_next in {"ˈ", "ˌ", "o", "e"}
                if prev_char and not prev_is_vowel and not next_is_vowel:
                    result.append(f"ᵊ{next_char}")
                    i += 2
                    continue
        result.append(char)
        i += 1
    return "".join(result)


def _apply_reduced_vowel_tuning(text: str) -> str:
    result: list[str] = []
    for i, char in enumerate(text):
        if char == "ɪ":
            prev_char = text[i - 1] if i > 0 else ""
            next_char = text[i + 1] if i + 1 < len(text) else ""
            if prev_char in {"ˈ", "ˌ"}:
                result.append(char)
                continue
            if "ˈ" in text[i + 1 :] and prev_char and prev_char not in _VOWELS:
                if next_char and next_char not in _VOWELS and next_char not in {"ˈ", "ˌ", "ŋ"}:
                    result.append("ə")
                    continue
        result.append(char)
    return "".join(result)


def espeak_ipa_to_misaki(ipa: str, *, british: bool, strategy: str = "espeak", word: str | None = None) -> str:
    strategy = normalize_strategy(strategy)
    result = ipa
    if strategy in _IPA_STRATEGIES:
        if strategy in _SYLLABIC_TUNED_STRATEGIES:
            rewrites = _IPA_REWRITES_VOWEL_SYLLABIC
        else:
            rewrites = _IPA_REWRITES_VOWEL if strategy in _VOWEL_TUNED_STRATEGIES else _IPA_REWRITES_BASE
        result = _apply_ipa_rewrites(result, rewrites)
    result = result.replace("\u0361", "^")
    for old, new in _FROM_ESPEAKS:
        result = result.replace(old, new)
    result = _swap_syllabic_markers(result)
    if british:
        result = result.replace("e^ə", "ɛː")
        result = result.replace("iə", "ɪə")
        result = result.replace("ə^ʊ", "Q")
    else:
        result = result.replace("o^ʊ", "O")
        result = result.replace("ɜːɹ", "ɜɹ")
        result = result.replace("ɜː", "ɜɹ")
        result = result.replace("ɪə", "iə")
        result = result.replace("ː", "")
    result = result.replace("^", "")
    if strategy in _VOWEL_TUNED_STRATEGIES:
        result = _apply_vowel_tuning(result)
    if strategy in _SYLLABIC_TUNED_STRATEGIES:
        result = _apply_syllabic_tuning(result)
    if strategy in _REDUCED_TUNED_STRATEGIES:
        result = _apply_reduced_vowel_tuning(result)
    if strategy in _PREFIX_STRESS_STRATEGIES:
        result = _apply_prefix_stress_rules(result, word)
    if strategy in _STRESS_TUNED_STRATEGIES:
        result = _apply_stress_normalization(result)
    if strategy in _FLAP_STRATEGIES:
        result = _apply_flap(result)
    return result


def fix_stress_placement(misaki_tokens: str) -> str:
    result: list[str] = []
    pending_stress: str | None = None

    for char in misaki_tokens:
        if char in {"ˈ", "ˌ"}:
            if pending_stress:
                result.append(pending_stress)
            pending_stress = char
        elif char in _VOWELS:
            if pending_stress:
                result.append(pending_stress)
                pending_stress = None
            result.append(char)
        else:
            result.append(char)

    if pending_stress:
        result.append(pending_stress)
    return "".join(result)
