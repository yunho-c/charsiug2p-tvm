#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostProcessStrategy {
    IpaFlapVowelReduced,
}

impl PostProcessStrategy {
    pub fn parse(value: &str) -> Result<Self, String> {
        let normalized = value.trim().to_lowercase().replace('_', "-");
        match normalized.as_str() {
            "ipa-flap-vowel-reduced" => Ok(Self::IpaFlapVowelReduced),
            _ => Err(format!("Unsupported post-process strategy: {value}")),
        }
    }

    pub fn variants() -> &'static [&'static str] {
        &["ipa-flap-vowel-reduced"]
    }
}

pub fn post_process(strategy: PostProcessStrategy, ipa: &str, british: bool) -> String {
    match strategy {
        PostProcessStrategy::IpaFlapVowelReduced => ipa_flap_vowel_reduced(ipa, british),
    }
}

const STRESS_PRIMARY: char = 'ˈ';
const STRESS_SECONDARY: char = 'ˌ';
const COMBINING_SYLLABIC: char = '\u{0329}';
const TIE_BAR: char = '\u{0361}';

const FROM_ESPEAKS: &[(&str, &str)] = &[
    ("ʔˌn\u{0329}", "tᵊn"),
    ("a^ɪ", "I"),
    ("a^ʊ", "W"),
    ("d^ʒ", "ʤ"),
    ("e^ɪ", "A"),
    ("t^ʃ", "ʧ"),
    ("ɔ^ɪ", "Y"),
    ("ə^l", "ᵊl"),
    ("ʔn", "tᵊn"),
    ("ʲO", "jO"),
    ("ʲQ", "jQ"),
    ("\u{0303}", ""),
    ("e", "A"),
    ("r", "ɹ"),
    ("x", "k"),
    ("ç", "k"),
    ("ɐ", "ə"),
    ("ɚ", "əɹ"),
    ("ɬ", "l"),
    ("ʔ", "t"),
    ("ʲ", ""),
];

const IPA_REWRITES_VOWEL: &[(&str, &str)] = &[
    ("tʃ", "t^ʃ"),
    ("dʒ", "d^ʒ"),
    ("oʊ", "o^ʊ"),
    ("əʊ", "ə^ʊ"),
    ("aɪ", "a^ɪ"),
    ("aʊ", "a^ʊ"),
    ("eɪ", "e^ɪ"),
    ("ɔɪ", "ɔ^ɪ"),
    ("ɝ", "ɜɹ"),
    ("ɫ", "l"),
    ("əl", "ə^l"),
    ("əɫ", "ə^l"),
];

fn ipa_flap_vowel_reduced(ipa: &str, british: bool) -> String {
    let mut result = apply_ipa_rewrites(ipa, IPA_REWRITES_VOWEL);
    result = result.replace(TIE_BAR, "^");
    result = apply_from_espeak(&result);
    result = swap_syllabic_markers(&result);
    if british {
        result = result.replace("e^ə", "ɛː");
        result = result.replace("iə", "ɪə");
        result = result.replace("ə^ʊ", "Q");
    } else {
        result = result.replace("o^ʊ", "O");
        result = result.replace("ɜːɹ", "ɜɹ");
        result = result.replace("ɜː", "ɜɹ");
        result = result.replace("ɪə", "iə");
        result = result.replace("ː", "");
    }
    result = result.replace("^", "");
    result = apply_vowel_tuning(&result);
    result = apply_reduced_vowel_tuning(&result);
    apply_flap(&result)
}

fn apply_ipa_rewrites(text: &str, rewrites: &[(&str, &str)]) -> String {
    let mut result = text.replace(TIE_BAR, "^");
    for (old, new) in rewrites {
        result = result.replace(old, new);
    }
    result
}

fn apply_from_espeak(text: &str) -> String {
    let mut result = text.to_string();
    for (old, new) in FROM_ESPEAKS {
        result = result.replace(old, new);
    }
    result
}

fn swap_syllabic_markers(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::new();
    let mut i = 0;
    while i < chars.len() {
        if i + 1 < chars.len() && chars[i + 1] == COMBINING_SYLLABIC {
            result.push('ᵊ');
            result.push(chars[i]);
            i += 2;
            continue;
        }
        if chars[i] != COMBINING_SYLLABIC {
            result.push(chars[i]);
        }
        i += 1;
    }
    result
}

fn apply_vowel_tuning(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::new();
    let mut pending_stress: Option<char> = None;
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch == STRESS_PRIMARY || ch == STRESS_SECONDARY {
            if let Some(stress) = pending_stress.take() {
                result.push(stress);
            }
            pending_stress = Some(ch);
            i += 1;
            continue;
        }
        if i + 1 < chars.len() && ((ch == 'ɜ' && chars[i + 1] == 'ɹ') || (ch == 'ə' && chars[i + 1] == 'ɹ')) {
            let stressed = pending_stress.is_some();
            if let Some(stress) = pending_stress.take() {
                result.push(stress);
            }
            if stressed {
                result.push_str("ɜɹ");
            } else {
                result.push_str("əɹ");
            }
            i += 2;
            continue;
        }
        if i + 1 < chars.len() && ch == 'ɪ' && (chars[i + 1] == 'd' || chars[i + 1] == 'z') {
            if let Some(stress) = pending_stress.take() {
                result.push(stress);
                result.push(ch);
                result.push(chars[i + 1]);
            } else {
                result.push('ᵻ');
                result.push(chars[i + 1]);
            }
            i += 2;
            continue;
        }
        if ch == 'ə' {
            if let Some(stress) = pending_stress.take() {
                result.push(stress);
                result.push('ʌ');
            } else {
                result.push(ch);
            }
            i += 1;
            continue;
        }
        if ch == 'ᵊ' {
            if let Some(stress) = pending_stress.take() {
                result.push(stress);
                result.push('ə');
            } else {
                result.push(ch);
            }
            i += 1;
            continue;
        }
        if pending_stress.is_some() && is_vowel(ch) {
            if let Some(stress) = pending_stress.take() {
                result.push(stress);
            }
        }
        result.push(ch);
        i += 1;
    }
    if let Some(stress) = pending_stress {
        result.push(stress);
    }
    result
}

fn apply_reduced_vowel_tuning(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::new();
    for i in 0..chars.len() {
        let ch = chars[i];
        if ch == 'ɪ' {
            let prev = if i > 0 { Some(chars[i - 1]) } else { None };
            let next = if i + 1 < chars.len() { Some(chars[i + 1]) } else { None };
            if prev == Some(STRESS_PRIMARY) || prev == Some(STRESS_SECONDARY) {
                result.push(ch);
                continue;
            }
            if let Some(prev_char) = prev {
                if !is_vowel(prev_char) {
                    let has_primary = chars[i + 1..].iter().any(|c| *c == STRESS_PRIMARY);
                    if has_primary {
                        if let Some(next_char) = next {
                            if !is_vowel(next_char)
                                && next_char != STRESS_PRIMARY
                                && next_char != STRESS_SECONDARY
                                && next_char != 'ŋ'
                            {
                                result.push('ə');
                                continue;
                            }
                        }
                    }
                }
            }
        }
        result.push(ch);
    }
    result
}

fn apply_flap(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::new();
    let mut i = 0;
    let mut last_vowel = false;
    while i < chars.len() {
        let ch = chars[i];
        if ch == STRESS_PRIMARY || ch == STRESS_SECONDARY {
            result.push(ch);
            i += 1;
            continue;
        }
        if ch == 't' && last_vowel {
            let mut j = i + 1;
            while j < chars.len() && (chars[j] == STRESS_PRIMARY || chars[j] == STRESS_SECONDARY) {
                j += 1;
            }
            if j < chars.len() && is_vowel(chars[j]) {
                result.push('ɾ');
                i += 1;
                last_vowel = false;
                continue;
            }
        }
        last_vowel = is_vowel(ch);
        result.push(ch);
        i += 1;
    }
    result
}

fn is_vowel(ch: char) -> bool {
    matches!(
        ch,
        'ɑ' | 'ɔ' | 'ɛ' | 'ɜ' | 'ɪ' | 'ʊ' | 'ʌ' | 'ə' | 'i' | 'u' | 'A' | 'I' | 'W' | 'Y'
            | 'O' | 'Q' | 'æ' | 'a' | 'ɒ' | 'ᵻ' | 'ᵊ'
    )
}

#[cfg(test)]
mod tests {
    use super::{post_process, PostProcessStrategy};

    #[test]
    fn flap_between_vowels() {
        let out = post_process(PostProcessStrategy::IpaFlapVowelReduced, "bʌtəɹ", false);
        assert_eq!(out, "bʌɾəɹ");
    }

    #[test]
    fn stressed_schwa_becomes_wedge() {
        let out = post_process(PostProcessStrategy::IpaFlapVowelReduced, "ˈəb", false);
        assert_eq!(out, "ˈʌb");
    }

    #[test]
    fn unstressed_ird_becomes_reduced() {
        let out = post_process(PostProcessStrategy::IpaFlapVowelReduced, "bɪd", false);
        assert_eq!(out, "bᵻd");
    }

    #[test]
    fn pretonic_reduction_applies() {
        let out = post_process(PostProcessStrategy::IpaFlapVowelReduced, "mɪnˈtɑl", false);
        assert_eq!(out, "məntˈɑl");
    }

    #[test]
    fn rhotic_defaults_to_schwa_when_unstressed() {
        let out = post_process(PostProcessStrategy::IpaFlapVowelReduced, "ɜɹ", false);
        assert_eq!(out, "əɹ");
    }
}
