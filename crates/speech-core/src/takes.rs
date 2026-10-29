//! Кеш тейків: аудіо кожного чанка глави та версії для A/B-порівняння.
//!
//! Для кожної глави поруч із `<index>.wav` лежить каталог `<index>/`:
//!
//! ```text
//! chapters/3.wav              # зібрана глава
//! chapters/3/takes.json       # маніфест юнітів і тейків
//! chapters/3/u0-t1.wav        # float32 WAV першого тейку юніта 0
//! chapters/3/u0-t2.wav
//! ```
//!
//! Юніт — це один чанк плану глави (зазвичай речення). Тейк зберігає сирі
//! float32-семпли до кросфейду та LUFS-нормалізації, тому глава з активних
//! тейків збирається байт-у-байт так само, як під час повного синтезу.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::projects::{now_rfc3339, ProjectSettings, PronunciationRule, Role};

/// Версія схеми маніфесту тейків.
pub const TAKES_SCHEMA_VERSION: u32 = 1;
/// Максимальна кількість юнітів у главі.
pub const MAX_UNITS: usize = 20_000;
/// Максимальна кількість тейків одного юніта.
pub const MAX_TAKES_PER_UNIT: usize = 50;
/// Максимальний id тейку (id монотонно зростає в межах юніта).
pub const MAX_TAKE_ID: u32 = 1_000_000;
/// Максимальна довжина тексту одного юніта.
pub const MAX_UNIT_TEXT_CHARS: usize = 20_000;

/// Маніфест тейків глави.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TakeManifest {
    pub version: u32,
    /// Відбиток тексту глави, налаштувань, ролей і словника вимови.
    pub fingerprint: String,
    pub units: Vec<UnitTakes>,
    #[serde(default = "now_rfc3339")]
    pub updated_at: String,
}

/// Тейки одного юніта (чанка) глави.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnitTakes {
    pub index: usize,
    /// Оригінальний текст для субтитрів (без вербалізації).
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// id активного тейку.
    pub active_take: u32,
    pub takes: Vec<Take>,
}

/// Одна версія аудіо юніта.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Take {
    pub id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    pub speed: f32,
    /// Текст, який фактично пішов у синтез (після вербалізації або override).
    pub spoken: String,
    pub duration_ms: u64,
    pub bytes: u64,
    /// SHA-256 float32 WAV; використовується для пошуку дублікатів.
    pub sha256: String,
    pub created_at: String,
}

impl TakeManifest {
    pub fn new(fingerprint: impl Into<String>, units: Vec<UnitTakes>) -> Self {
        Self {
            version: TAKES_SCHEMA_VERSION,
            fingerprint: fingerprint.into(),
            units,
            updated_at: now_rfc3339(),
        }
    }

    pub fn touch(&mut self) {
        self.updated_at = now_rfc3339();
    }

    pub fn unit(&self, index: usize) -> Option<&UnitTakes> {
        self.units.iter().find(|unit| unit.index == index)
    }

    pub fn unit_mut(&mut self, index: usize) -> Option<&mut UnitTakes> {
        self.units.iter_mut().find(|unit| unit.index == index)
    }

    /// Перевірити структуру маніфесту перед збереженням/використанням.
    pub fn validate(&self) -> Result<()> {
        if self.version != TAKES_SCHEMA_VERSION {
            bail!(
                "unsupported takes manifest version: {} (expected {})",
                self.version,
                TAKES_SCHEMA_VERSION
            );
        }
        if self.fingerprint.trim().is_empty() {
            bail!("takes fingerprint must not be empty");
        }
        if self.units.is_empty() {
            bail!("takes manifest has no units");
        }
        if self.units.len() > MAX_UNITS {
            bail!("too many units: {}", self.units.len());
        }
        for (position, unit) in self.units.iter().enumerate() {
            unit.validate(position)?;
        }
        Ok(())
    }
}

impl UnitTakes {
    fn validate(&self, position: usize) -> Result<()> {
        if self.index != position {
            bail!(
                "unit index mismatch: expected {position}, got {}",
                self.index
            );
        }
        if self.text.trim().is_empty() {
            bail!("unit {} text must not be empty", self.index);
        }
        if self.text.chars().count() > MAX_UNIT_TEXT_CHARS {
            bail!("unit {} text is too long", self.index);
        }
        if self.takes.is_empty() {
            bail!("unit {} has no takes", self.index);
        }
        if self.takes.len() > MAX_TAKES_PER_UNIT {
            bail!("unit {} has too many takes", self.index);
        }
        let mut ids = std::collections::HashSet::with_capacity(self.takes.len());
        for take in &self.takes {
            take.validate(self.index)?;
            if !ids.insert(take.id) {
                bail!("unit {} has duplicate take id {}", self.index, take.id);
            }
        }
        if !self.takes.iter().any(|take| take.id == self.active_take) {
            bail!(
                "unit {} active take {} does not exist",
                self.index,
                self.active_take
            );
        }
        Ok(())
    }

    pub fn take(&self, id: u32) -> Option<&Take> {
        self.takes.iter().find(|take| take.id == id)
    }

    pub fn active(&self) -> Option<&Take> {
        self.take(self.active_take)
    }

    pub fn next_take_id(&self) -> Result<u32> {
        let next = self
            .takes
            .iter()
            .map(|take| take.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        if next > MAX_TAKE_ID {
            bail!("unit {} has too many takes", self.index);
        }
        Ok(next)
    }
}

impl Take {
    fn validate(&self, unit: usize) -> Result<()> {
        if self.id == 0 || self.id > MAX_TAKE_ID {
            bail!("unit {unit} has invalid take id {}", self.id);
        }
        if !self.speed.is_finite() || !(0.5..=2.0).contains(&self.speed) {
            bail!("unit {unit} take {} has invalid speed", self.id);
        }
        if self.spoken.trim().is_empty() {
            bail!("unit {unit} take {} has empty spoken text", self.id);
        }
        if self.sha256.len() != 64 || !self.sha256.chars().all(|ch| ch.is_ascii_hexdigit()) {
            bail!("unit {unit} take {} has invalid sha256", self.id);
        }
        Ok(())
    }
}

/// Джерело відбитка: усе, що впливає на план чанків і голоси.
#[derive(Serialize)]
struct FingerprintSource<'a> {
    text: &'a str,
    settings: &'a ProjectSettings,
    roles: &'a [Role],
    pronunciation: &'a [PronunciationRule],
}

/// SHA-256 відбиток глави для інвалідації кешу тейків.
pub fn chapter_fingerprint(
    text: &str,
    settings: &ProjectSettings,
    roles: &[Role],
    pronunciation: &[PronunciationRule],
) -> String {
    let source = FingerprintSource {
        text,
        settings,
        roles,
        pronunciation,
    };
    match serde_json::to_vec(&source) {
        Ok(bytes) => sha256_hex(&bytes),
        Err(error) => {
            tracing::error!(error = %error, "cannot serialize takes fingerprint");
            String::new()
        }
    }
}

/// SHA-256 у нижньому hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::Chapter;

    fn settings() -> ProjectSettings {
        ProjectSettings::default()
    }

    fn take(id: u32, spoken: &str) -> Take {
        Take {
            id,
            voice: Some("Speaker_0".into()),
            speed: 1.0,
            spoken: spoken.into(),
            duration_ms: 1_000,
            bytes: 48_044,
            sha256: sha256_hex(spoken.as_bytes()),
            created_at: now_rfc3339(),
        }
    }

    fn manifest() -> TakeManifest {
        TakeManifest::new(
            "fingerprint",
            vec![UnitTakes {
                index: 0,
                text: "Привіт, світе!".into(),
                role: None,
                active_take: 1,
                takes: vec![take(1, "привіт, світе")],
            }],
        )
    }

    #[test]
    fn manifest_roundtrip_and_validation() {
        let manifest = manifest();
        manifest.validate().unwrap();
        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: TakeManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, manifest);
        assert_eq!(parsed.unit(0).unwrap().active().unwrap().id, 1);
        assert_eq!(parsed.unit(0).unwrap().next_take_id().unwrap(), 2);
    }

    #[test]
    fn rejects_broken_manifests() {
        let mut broken = manifest();
        broken.units[0].active_take = 9;
        assert!(broken.validate().is_err());

        let mut broken = manifest();
        broken.units[0].takes.push(take(1, "дубль"));
        assert!(broken.validate().is_err());

        let mut broken = manifest();
        broken.units[0].index = 4;
        assert!(broken.validate().is_err());

        let mut broken = manifest();
        broken.units[0].takes[0].sha256 = "xyz".into();
        assert!(broken.validate().is_err());

        let mut broken = manifest();
        broken.version = TAKES_SCHEMA_VERSION + 1;
        assert!(broken.validate().is_err());
    }

    #[test]
    fn fingerprint_tracks_plan_inputs() {
        let chapter = Chapter::new(0, "Розділ", "Текст глави.");
        let roles = vec![Role::with_voice("Іван", "Speaker_1")];
        let base = chapter_fingerprint(&chapter.text, &settings(), &roles, &[]);
        assert_eq!(base.len(), 64);

        assert_ne!(
            base,
            chapter_fingerprint("Інший текст.", &settings(), &roles, &[])
        );
        assert_ne!(
            base,
            chapter_fingerprint(
                &chapter.text,
                &settings(),
                &[Role::with_voice("Іван", "Speaker_3")],
                &[]
            )
        );
        let mut changed = settings();
        changed.speed = 1.1;
        assert_ne!(
            base,
            chapter_fingerprint(&chapter.text, &changed, &roles, &[])
        );
        assert_ne!(
            base,
            chapter_fingerprint(
                &chapter.text,
                &settings(),
                &roles,
                &[PronunciationRule {
                    pattern: "ІТ".into(),
                    replacement: "ай-ті".into(),
                    regex: false,
                    case_sensitive: true,
                }]
            )
        );
    }

    #[test]
    fn sha256_is_stable() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
