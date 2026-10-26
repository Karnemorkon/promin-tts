//! Словники вимови: заміна підрядків/regex до вербалізації.
//!
//! Правила застосовуються послідовно. Літеральні правила типово
//! регістронезалежні; regex-правила підтримують групи захоплення (`$1`).
//! Проєктні правила перекривають глобальні з тим самим `pattern`
//! (див. [`merge_rules`]).

use std::borrow::Cow;

use regex::{NoExpand, Regex, RegexBuilder};

/// Максимальна кількість правил у словнику.
pub const MAX_RULES: usize = 1_000;
/// Максимальна довжина `pattern` (символів).
pub const MAX_PATTERN_CHARS: usize = 1_000;
/// Максимальна довжина `replacement` (символів).
pub const MAX_REPLACEMENT_CHARS: usize = 1_000;

/// Правило вимови.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    pub pattern: String,
    pub replacement: String,
    /// `pattern` — регулярний вираз (`regex` crate).
    pub regex: bool,
    /// Враховувати регістр.
    pub case_sensitive: bool,
}

impl Rule {
    pub fn literal(pattern: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            replacement: replacement.into(),
            regex: false,
            case_sensitive: false,
        }
    }

    pub fn regex(pattern: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            replacement: replacement.into(),
            regex: true,
            case_sensitive: false,
        }
    }

    fn validate(&self) -> Result<(), PronunciationError> {
        if self.pattern.trim().is_empty() {
            return Err(PronunciationError::EmptyPattern);
        }
        if self.pattern.chars().count() > MAX_PATTERN_CHARS {
            return Err(PronunciationError::PatternTooLong);
        }
        if self.replacement.chars().count() > MAX_REPLACEMENT_CHARS {
            return Err(PronunciationError::ReplacementTooLong);
        }
        Ok(())
    }
}

/// Помилка компіляції словника.
#[derive(Debug, thiserror::Error)]
pub enum PronunciationError {
    #[error("pronunciation pattern must not be empty")]
    EmptyPattern,
    #[error("pronunciation pattern is too long (max {MAX_PATTERN_CHARS} chars)")]
    PatternTooLong,
    #[error("pronunciation replacement is too long (max {MAX_REPLACEMENT_CHARS} chars)")]
    ReplacementTooLong,
    #[error("too many pronunciation rules: {count} (max {MAX_RULES})")]
    TooManyRules { count: usize },
    #[error("invalid pronunciation regex {pattern:?}: {source}")]
    InvalidRegex {
        pattern: String,
        #[source]
        source: regex::Error,
    },
}

#[derive(Debug)]
enum CompiledRule {
    Literal {
        needle: String,
        case_sensitive: bool,
        replacement: String,
    },
    Regex {
        regex: Regex,
        replacement: String,
        /// Чи розкривати `$1`/`${name}` у заміні.
        expand: bool,
    },
}

impl CompiledRule {
    fn compile(rule: &Rule) -> Result<Self, PronunciationError> {
        rule.validate()?;
        if rule.regex {
            let regex = RegexBuilder::new(&rule.pattern)
                .case_insensitive(!rule.case_sensitive)
                .build()
                .map_err(|source| PronunciationError::InvalidRegex {
                    pattern: rule.pattern.clone(),
                    source,
                })?;
            return Ok(Self::Regex {
                regex,
                replacement: rule.replacement.clone(),
                expand: true,
            });
        }
        if rule.case_sensitive {
            return Ok(Self::Literal {
                needle: rule.pattern.clone(),
                case_sensitive: true,
                replacement: rule.replacement.clone(),
            });
        }
        let regex = RegexBuilder::new(&regex::escape(&rule.pattern))
            .case_insensitive(true)
            .build()
            .map_err(|source| PronunciationError::InvalidRegex {
                pattern: rule.pattern.clone(),
                source,
            })?;
        Ok(Self::Regex {
            regex,
            replacement: rule.replacement.clone(),
            expand: false,
        })
    }

    fn apply(&self, text: &str) -> Option<String> {
        match self {
            Self::Literal {
                needle,
                case_sensitive: true,
                replacement,
            } => text
                .contains(needle)
                .then(|| text.replace(needle, replacement)),
            Self::Literal { .. } => unreachable!("case-insensitive literals compile as regex"),
            Self::Regex {
                regex,
                replacement,
                expand,
            } => regex.is_match(text).then(|| {
                if *expand {
                    regex.replace_all(text, replacement.as_str()).into_owned()
                } else {
                    regex.replace_all(text, NoExpand(replacement)).into_owned()
                }
            }),
        }
    }
}

/// Скомпільований словник вимови.
#[derive(Debug, Default)]
pub struct Pronouncer {
    rules: Vec<CompiledRule>,
}

impl Pronouncer {
    /// Скомпілювати правила; порожній список — валідний «порожній» словник.
    pub fn new(rules: impl IntoIterator<Item = Rule>) -> Result<Self, PronunciationError> {
        let rules: Vec<Rule> = rules.into_iter().collect();
        if rules.len() > MAX_RULES {
            return Err(PronunciationError::TooManyRules { count: rules.len() });
        }
        let compiled = rules
            .iter()
            .map(CompiledRule::compile)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { rules: compiled })
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Застосувати всі правила послідовно.
    pub fn apply<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if self.rules.is_empty() {
            return Cow::Borrowed(text);
        }
        let mut current = text.to_string();
        for rule in &self.rules {
            if let Some(next) = rule.apply(&current) {
                current = next;
            }
        }
        Cow::Owned(current)
    }
}

/// Злити глобальні та проєктні правила: проєктні перекривають глобальні
/// з тим самим `pattern`.
pub fn merge_rules(global: &[Rule], project: &[Rule]) -> Vec<Rule> {
    let overridden: std::collections::HashSet<&str> =
        project.iter().map(|rule| rule.pattern.as_str()).collect();
    global
        .iter()
        .filter(|rule| !overridden.contains(rule.pattern.as_str()))
        .cloned()
        .chain(project.iter().cloned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_rules_are_case_insensitive_by_default() {
        let pronouncer = Pronouncer::new(vec![Rule::literal("промінь", "прóмінь")]).unwrap();
        assert_eq!(pronouncer.apply("ПРОМІНЬ і промінь"), "прóмінь і прóмінь");
    }

    #[test]
    fn literal_rules_can_be_case_sensitive() {
        let mut rule = Rule::literal("ІТ", "ай-ті");
        rule.case_sensitive = true;
        let pronouncer = Pronouncer::new(vec![rule]).unwrap();
        assert_eq!(pronouncer.apply("ІТ та іт"), "ай-ті та іт");
    }

    #[test]
    fn literal_replacement_does_not_expand_groups() {
        let pronouncer = Pronouncer::new(vec![Rule::literal("ціна", "$1")]).unwrap();
        assert_eq!(pronouncer.apply("ціна"), "$1");
    }

    #[test]
    fn regex_rules_support_capture_groups() {
        let pronouncer =
            Pronouncer::new(vec![Rule::regex(r"(\d+)\s*км", "$1 кілометрів")]).unwrap();
        assert_eq!(
            pronouncer.apply("5км і 12 км"),
            "5 кілометрів і 12 кілометрів"
        );
    }

    #[test]
    fn rules_apply_in_order() {
        let pronouncer =
            Pronouncer::new(vec![Rule::literal("А", "Б"), Rule::literal("Б", "В")]).unwrap();
        assert_eq!(pronouncer.apply("А"), "В");
    }

    #[test]
    fn merge_gives_project_rules_priority() {
        let global = vec![
            Rule::literal("лікар", "лікар"),
            Rule::literal("мова", "мова"),
        ];
        let project = vec![Rule::literal("лікар", "лікарка")];
        let merged = merge_rules(&global, &project);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].pattern, "мова");
        assert_eq!(merged[1].replacement, "лікарка");
        let pronouncer = Pronouncer::new(merged).unwrap();
        assert_eq!(pronouncer.apply("лікар і мова"), "лікарка і мова");
    }

    #[test]
    fn rejects_invalid_regex_and_limits() {
        assert!(matches!(
            Pronouncer::new(vec![Rule::regex("(", "x")]),
            Err(PronunciationError::InvalidRegex { .. })
        ));
        assert!(matches!(
            Pronouncer::new(vec![Rule::literal(" ", "x")]),
            Err(PronunciationError::EmptyPattern)
        ));
        let rules = (0..=MAX_RULES)
            .map(|index| Rule::literal(format!("p{index}"), "x"))
            .collect::<Vec<_>>();
        assert!(matches!(
            Pronouncer::new(rules),
            Err(PronunciationError::TooManyRules { .. })
        ));
    }

    #[test]
    fn empty_pronouncer_is_borrowed() {
        let pronouncer = Pronouncer::new(Vec::new()).unwrap();
        assert!(pronouncer.is_empty());
        assert!(matches!(pronouncer.apply("текст"), Cow::Borrowed("текст")));
    }
}
