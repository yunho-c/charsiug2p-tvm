from __future__ import annotations

SUPPORTED_STRATEGIES = {"espeak", "ipa", "ipa-flap"}

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

_IPA_REWRITES = sorted(
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


def normalize_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower().replace("_", "-")
    if normalized in {"ipa+flap", "ipa+flaps"}:
        normalized = "ipa-flap"
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


def _apply_ipa_rewrites(text: str) -> str:
    text = text.replace("\u0361", "^")
    for old, new in _IPA_REWRITES:
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


def espeak_ipa_to_misaki(ipa: str, *, british: bool, strategy: str = "espeak") -> str:
    strategy = normalize_strategy(strategy)
    result = ipa
    if strategy in {"ipa", "ipa-flap"}:
        result = _apply_ipa_rewrites(result)
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
    if strategy == "ipa-flap":
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
