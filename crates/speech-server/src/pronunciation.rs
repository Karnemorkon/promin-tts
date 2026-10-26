//! Глобальний словник вимови: файл, валідація та компільований кеш.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use speech_core::projects::{PronunciationRule, MAX_PRONUNCIATION_RULES};
use speech_text::pronunciation::{merge_rules, Pronouncer, Rule};

/// Вміст `<data>/pronunciation.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PronunciationFile {
    #[serde(default)]
    pub rules: Vec<PronunciationRule>,
}

/// Тіло `PUT /v1/pronunciation`: `{"rules": [...]}` або голий масив.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PronunciationPayload {
    Wrapped(PronunciationFile),
    Bare(Vec<PronunciationRule>),
}

impl PronunciationPayload {
    pub fn into_rules(self) -> Vec<PronunciationRule> {
        match self {
            Self::Wrapped(file) => file.rules,
            Self::Bare(rules) => rules,
        }
    }
}

/// Помилка збереження словника.
#[derive(Debug, thiserror::Error)]
pub enum PronunciationStoreError {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] anyhow::Error),
}

/// Словник вимови з атомарним файловим збереженням.
pub struct PronunciationStore {
    path: PathBuf,
    rules: RwLock<Vec<PronunciationRule>>,
    pronouncer: RwLock<Arc<Pronouncer>>,
}

impl PronunciationStore {
    /// Завантажити словник; відсутній файл — порожній словник.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let rules = if path.is_file() {
            let bytes = std::fs::read(&path)
                .with_context(|| format!("read pronunciation file {}", path.display()))?;
            let payload: PronunciationPayload = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse pronunciation file {}", path.display()))?;
            payload.into_rules()
        } else {
            Vec::new()
        };
        let pronouncer = compile_rules(&rules)
            .map_err(|error| anyhow::anyhow!("invalid pronunciation file: {error}"))?;
        Ok(Self {
            path,
            rules: RwLock::new(rules),
            pronouncer: RwLock::new(Arc::new(pronouncer)),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn rules(&self) -> Vec<PronunciationRule> {
        self.rules
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn rule_count(&self) -> usize {
        self.rules
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Поточний скомпільований глобальний словник.
    pub fn pronouncer(&self) -> Arc<Pronouncer> {
        Arc::clone(
            &self
                .pronouncer
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    /// Замінити словник: спершу валідація/компіляція, потім атомарний запис.
    pub fn replace(
        &self,
        rules: Vec<PronunciationRule>,
    ) -> Result<Vec<PronunciationRule>, PronunciationStoreError> {
        let pronouncer = compile_rules(&rules).map_err(|error| {
            PronunciationStoreError::Invalid(format!("invalid pronunciation rules: {error}"))
        })?;
        let file = PronunciationFile {
            rules: rules.clone(),
        };
        let json = serde_json::to_vec_pretty(&file)
            .map_err(|error| PronunciationStoreError::Io(error.into()))?;
        write_atomic(&self.path, &json).map_err(PronunciationStoreError::Io)?;
        *self
            .rules
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = rules.clone();
        *self
            .pronouncer
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Arc::new(pronouncer);
        Ok(rules)
    }

    /// Глобальні + проєктні правила (проєктні перекривають глобальні).
    /// Використовується генерацією глав проєкту.
    #[allow(dead_code)]
    pub fn merged_pronouncer(
        &self,
        project_rules: &[PronunciationRule],
    ) -> Result<Arc<Pronouncer>, PronunciationStoreError> {
        let global = self.rules();
        let global: Vec<Rule> = global.iter().map(to_rule).collect();
        let project: Vec<Rule> = project_rules.iter().map(to_rule).collect();
        let merged = merge_rules(&global, &project);
        Pronouncer::new(merged)
            .map(Arc::new)
            .map_err(|error| PronunciationStoreError::Invalid(error.to_string()))
    }
}

/// Конвертувати правило з моделі проєкту в правило рушія.
pub fn to_rule(rule: &PronunciationRule) -> Rule {
    Rule {
        pattern: rule.pattern.clone(),
        replacement: rule.replacement.clone(),
        regex: rule.regex,
        case_sensitive: rule.case_sensitive,
    }
}

fn compile_rules(rules: &[PronunciationRule]) -> Result<Pronouncer, String> {
    if rules.len() > MAX_PRONUNCIATION_RULES {
        return Err(format!(
            "too many rules: {} (max {MAX_PRONUNCIATION_RULES})",
            rules.len()
        ));
    }
    for rule in rules {
        rule.validate().map_err(|error| error.to_string())?;
    }
    Pronouncer::new(rules.iter().map(to_rule)).map_err(|error| error.to_string())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create directory {}", parent.display()))?;
        }
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("pronunciation.json");
    let tmp = path.with_file_name(format!(".{file_name}.tmp-{}", uuid::Uuid::new_v4()));
    std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path() -> PathBuf {
        std::env::temp_dir().join(format!(
            "promin-pronunciation-test-{}.json",
            uuid::Uuid::new_v4()
        ))
    }

    fn rule(pattern: &str, replacement: &str) -> PronunciationRule {
        PronunciationRule {
            pattern: pattern.to_string(),
            replacement: replacement.to_string(),
            regex: false,
            case_sensitive: false,
        }
    }

    #[test]
    fn missing_file_is_empty_and_replace_persists() {
        let path = temp_path();
        let store = PronunciationStore::load(&path).unwrap();
        assert!(store.rules().is_empty());

        store.replace(vec![rule("промінь", "прóмінь")]).unwrap();
        assert_eq!(store.rules().len(), 1);
        assert_eq!(store.pronouncer().apply("ПРОМІНЬ"), "прóмінь");

        let reloaded = PronunciationStore::load(&path).unwrap();
        assert_eq!(reloaded.rules().len(), 1);
        assert_eq!(reloaded.pronouncer().apply("промінь"), "прóмінь");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_rules_do_not_overwrite_file() {
        let path = temp_path();
        let store = PronunciationStore::load(&path).unwrap();
        store.replace(vec![rule("a", "b")]).unwrap();
        let error = store
            .replace(vec![PronunciationRule {
                pattern: "(".to_string(),
                replacement: "x".to_string(),
                regex: true,
                case_sensitive: false,
            }])
            .unwrap_err();
        assert!(matches!(error, PronunciationStoreError::Invalid(_)));
        assert_eq!(store.rules(), vec![rule("a", "b")]);
        let reloaded = PronunciationStore::load(&path).unwrap();
        assert_eq!(reloaded.rules(), vec![rule("a", "b")]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn project_rules_override_global() {
        let path = temp_path();
        let store = PronunciationStore::load(&path).unwrap();
        store
            .replace(vec![rule("лікар", "лікар"), rule("мова", "мова")])
            .unwrap();
        let pronouncer = store
            .merged_pronouncer(&[rule("лікар", "лікарка")])
            .unwrap();
        assert_eq!(pronouncer.apply("лікар і мова"), "лікарка і мова");
        assert_eq!(store.rules().len(), 2);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn accepts_bare_array_and_wrapped_object() {
        let wrapped: PronunciationPayload =
            serde_json::from_str(r#"{"rules":[{"pattern":"a","replacement":"b"}]}"#).unwrap();
        assert_eq!(wrapped.into_rules().len(), 1);
        let bare: PronunciationPayload =
            serde_json::from_str(r#"[{"pattern":"a","replacement":"b"}]"#).unwrap();
        assert_eq!(bare.into_rules().len(), 1);
    }
}
