use std::fmt;

pub const DEFAULT_MAX_INPUT_BYTES: usize = 64;
pub const DEFAULT_MAX_OUTPUT_LEN: usize = 128;

#[derive(Debug, Clone)]
pub struct G2pConfig {
    pub max_input_bytes: usize,
    pub max_output_len: usize,
    pub space_after_colon: bool,
}

impl Default for G2pConfig {
    fn default() -> Self {
        Self {
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            max_output_len: DEFAULT_MAX_OUTPUT_LEN,
            space_after_colon: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum G2pError {
    InputTooLong { max_input_bytes: usize, examples: Vec<(String, usize)> },
}

impl fmt::Display for G2pError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            G2pError::InputTooLong { max_input_bytes, examples } => {
                let formatted = examples
                    .iter()
                    .take(5)
                    .map(|(word, size)| format!("{word}={size}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "Input exceeds max_input_bytes={max_input_bytes} (examples: {formatted}). Increase the bound or filter long inputs."
                )
            }
        }
    }
}

impl std::error::Error for G2pError {}

pub fn add_language_prefix(words: &[String], lang: &str, space_after_colon: bool) -> Vec<String> {
    let prefix = if space_after_colon {
        format!("<{lang}>: ")
    } else {
        format!("<{lang}>:")
    };
    words.iter().map(|word| format!("{prefix}{word}")).collect()
}

pub fn validate_input_bytes(
    words: &[String],
    prefixed_words: &[String],
    max_input_bytes: usize,
) -> Result<(), G2pError> {
    let mut oversized = Vec::new();
    for (word, prefixed) in words.iter().zip(prefixed_words.iter()) {
        let size = prefixed.as_bytes().len();
        if size > max_input_bytes {
            oversized.push((word.clone(), size));
        }
    }
    if !oversized.is_empty() {
        return Err(G2pError::InputTooLong {
            max_input_bytes,
            examples: oversized,
        });
    }
    Ok(())
}

pub fn prepare_prefixed_words(
    words: &[String],
    lang: &str,
    space_after_colon: bool,
    max_input_bytes: usize,
) -> Result<Vec<String>, G2pError> {
    let prefixed = add_language_prefix(words, lang, space_after_colon);
    validate_input_bytes(words, &prefixed, max_input_bytes)?;
    Ok(prefixed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_without_space() {
        let words = vec!["Char".to_string(), "siu".to_string()];
        let prefixed = add_language_prefix(&words, "eng-us", false);
        assert_eq!(prefixed, vec!["<eng-us>:Char", "<eng-us>:siu"]);
    }

    #[test]
    fn prefix_with_space() {
        let words = vec!["Char".to_string()];
        let prefixed = add_language_prefix(&words, "eng-us", true);
        assert_eq!(prefixed, vec!["<eng-us>: Char"]);
    }

    #[test]
    fn validate_input_bytes_rejects_long() {
        let words = vec!["too-long".to_string()];
        let prefixed = vec!["<eng-us>:too-long".to_string()];
        let err = validate_input_bytes(&words, &prefixed, 4).unwrap_err();
        match err {
            G2pError::InputTooLong { max_input_bytes, examples } => {
                assert_eq!(max_input_bytes, 4);
                assert_eq!(examples[0].0, "too-long");
            }
        }
    }
}
