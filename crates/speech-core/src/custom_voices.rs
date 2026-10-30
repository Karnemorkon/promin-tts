//! Кастомні (клоновані) голоси: `<dir>/<name>.npy` (float32[256]).
//!
//! Порт `custom_voices.cpp` з holos-tts-core, сумісний з numpy
//! (`np.save`/`np.load`) і HolosTTS-hybrid. Каталог сканується при кожному
//! зверненні: `.npy`, покладені у volume, зʼявляються без рестарту.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Розмірність ембеддінгу голосу (вхід `voice` ONNX-генератора).
pub const VOICE_DIM: usize = 256;

const NPY_MAGIC: &[u8] = b"\x93NUMPY";

/// Помилки сховища кастомних голосів.
#[derive(Debug, thiserror::Error)]
pub enum CustomVoiceError {
    #[error("custom voices are disabled")]
    Disabled,
    #[error("invalid voice name: {0:?}")]
    InvalidName(String),
    #[error("embedding must be {VOICE_DIM} floats, got {0}")]
    WrongDim(usize),
    #[error("unknown custom voice: {0}")]
    Unknown(String),
    #[error("npy: {0}")]
    Npy(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl CustomVoiceError {
    /// Помилка запиту клієнта (HTTP 400) чи внутрішня.
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidName(_) | Self::WrongDim(_) | Self::Npy(_)
        )
    }
}

/// Назва голосу: `^[\w \-]{1,40}$` (Unicode, як PCRE2_UCP у C++).
///
/// Дозволені літери/цифри/`_`, пробіл і дефіс; довжина — 1..=40 кодпойнтів.
pub fn valid_name(name: &str) -> bool {
    let mut count = 0usize;
    for ch in name.chars() {
        if !(ch.is_alphanumeric() || ch == '_' || ch == ' ' || ch == '-') {
            return false;
        }
        count += 1;
        if count > 40 {
            return false;
        }
    }
    count > 0
}

fn le_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Перше ціле з `'shape': (N,)` у numpy-заголовку (як C++ `parse_shape_n`).
fn parse_shape_n(header: &str) -> Result<usize, CustomVoiceError> {
    let shape = header
        .find("'shape'")
        .ok_or_else(|| CustomVoiceError::Npy("missing 'shape'".into()))?;
    let open = header[shape..]
        .find('(')
        .map(|offset| shape + offset)
        .ok_or_else(|| CustomVoiceError::Npy("malformed 'shape'".into()))?;
    let rest = &header[open + 1..];
    let digits: String = rest
        .trim_start_matches([' ', '('])
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        return Err(CustomVoiceError::Npy("malformed 'shape'".into()));
    }
    let n: usize = digits
        .parse()
        .map_err(|_| CustomVoiceError::Npy("shape overflow".into()))?;
    let tail = &rest[rest.find(&digits).expect("digits came from rest") + digits.len()..];
    let tail = tail.trim_start_matches(' ');
    let tail = tail
        .strip_prefix(',')
        .unwrap_or(tail)
        .trim_start_matches(' ');
    if !tail.starts_with(')') {
        return Err(CustomVoiceError::Npy("expected 1-D array".into()));
    }
    Ok(n)
}

/// numpy `.npy` (1-D float32, little-endian) -> `Vec<f32>`.
pub fn load_npy_f32(path: impl AsRef<Path>) -> Result<Vec<f32>, CustomVoiceError> {
    let bytes = fs::read(path)?;
    load_npy_f32_bytes(&bytes)
}

/// Те саме з байтів (для тестів і майбутніх upload-ів).
pub fn load_npy_f32_bytes(bytes: &[u8]) -> Result<Vec<f32>, CustomVoiceError> {
    if bytes.len() < 10 || !bytes.starts_with(NPY_MAGIC) {
        return Err(CustomVoiceError::Npy("invalid magic".into()));
    }
    let major = bytes[6];
    let (header_len, header_offset) = match major {
        1 => (usize::from(le_u16(&bytes[8..10])), 10usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(CustomVoiceError::Npy("truncated header".into()));
            }
            (le_u32(&bytes[8..12]) as usize, 12usize)
        }
        _ => return Err(CustomVoiceError::Npy("unsupported version".into())),
    };
    let header_end = header_offset
        .checked_add(header_len)
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| CustomVoiceError::Npy("truncated header".into()))?;
    let header = std::str::from_utf8(&bytes[header_offset..header_end])
        .map_err(|_| CustomVoiceError::Npy("header is not UTF-8".into()))?;
    if !header.contains("<f4") && !header.contains("=f4") {
        return Err(CustomVoiceError::Npy(
            "only little-endian float32 is supported".into(),
        ));
    }
    if header.contains("'fortran_order': True") {
        return Err(CustomVoiceError::Npy(
            "fortran_order is not supported".into(),
        ));
    }

    let n = parse_shape_n(header)?;
    let data_len = n
        .checked_mul(4)
        .ok_or_else(|| CustomVoiceError::Npy("shape overflow".into()))?;
    if bytes.len() < header_end + data_len {
        return Err(CustomVoiceError::Npy("truncated data".into()));
    }
    Ok(bytes[header_end..header_end + data_len]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

/// `Vec<f32>` -> numpy `.npy` (версія 1.0, 1-D little-endian float32).
pub fn save_npy_f32(path: impl AsRef<Path>, values: &[f32]) -> Result<(), CustomVoiceError> {
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        values.len()
    );
    // Заголовок разом із newline вирівнюється до 64 байт (як np.save).
    let mut total = 10 + header.len() + 1;
    while total % 64 != 0 {
        header.push(' ');
        total += 1;
    }
    header.push('\n');
    let header_len = u16::try_from(header.len())
        .map_err(|_| CustomVoiceError::Npy("header is too long".into()))?;

    let mut out = Vec::with_capacity(10 + header.len() + values.len() * 4);
    out.extend_from_slice(NPY_MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, out)?;
    Ok(())
}

/// Сховище кастомних голосів у каталозі (порожній шлях -> вимкнено).
pub struct CustomVoices {
    dir: PathBuf,
    map: Mutex<BTreeMap<String, Vec<f32>>>,
}

impl CustomVoices {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            map: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn enabled(&self) -> bool {
        !self.dir.as_os_str().is_empty()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path_for(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{name}.npy"))
    }

    /// Підхоплює нові `.npy` з каталогу; пошкоджені та не-256 ігноруються.
    fn rescan(&self, map: &mut BTreeMap<String, Vec<f32>>) {
        if !self.enabled() || !self.dir.is_dir() {
            return;
        }
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return;
        };
        let mut found = BTreeMap::new();
        for entry in entries.flatten() {
            if !entry.path().is_file() {
                continue;
            }
            let Some(filename) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let Some(name) = filename.strip_suffix(".npy") else {
                continue;
            };
            if let Some(cached) = map.get(name) {
                found.insert(name.to_string(), cached.clone());
                continue;
            }
            if let Ok(vector) = load_npy_f32(entry.path()) {
                if vector.len() == VOICE_DIM {
                    found.insert(name.to_string(), vector);
                }
            }
        }
        *map = found;
    }

    pub fn names(&self) -> Vec<String> {
        let mut map = self
            .map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.rescan(&mut map);
        map.keys().cloned().collect()
    }

    pub fn has(&self, name: &str) -> bool {
        let mut map = self
            .map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.rescan(&mut map);
        map.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<Vec<f32>, CustomVoiceError> {
        let mut map = self
            .map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.rescan(&mut map);
        map.get(name)
            .cloned()
            .ok_or_else(|| CustomVoiceError::Unknown(name.to_string()))
    }

    pub fn save(&self, name: &str, embedding: &[f32]) -> Result<(), CustomVoiceError> {
        if !self.enabled() {
            return Err(CustomVoiceError::Disabled);
        }
        if !valid_name(name) {
            return Err(CustomVoiceError::InvalidName(name.to_string()));
        }
        if embedding.len() != VOICE_DIM {
            return Err(CustomVoiceError::WrongDim(embedding.len()));
        }
        fs::create_dir_all(&self.dir)?;
        let path = self.path_for(name);
        let tmp = path.with_extension("npy.tmp");
        save_npy_f32(&tmp, embedding)?;
        if let Err(error) = fs::rename(&tmp, &path) {
            let _ = fs::remove_file(&tmp);
            return Err(error.into());
        }
        let mut map = self
            .map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        map.insert(name.to_string(), embedding.to_vec());
        Ok(())
    }

    pub fn remove(&self, name: &str) -> Result<bool, CustomVoiceError> {
        if !self.enabled() {
            return Ok(false);
        }
        let mut map = self
            .map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.rescan(&mut map);
        if map.remove(name).is_none() {
            return Ok(false);
        }
        fs::remove_file(self.path_for(name))?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "promin-voices-{tag}-{}-{stamp}",
            std::process::id()
        ))
    }

    #[test]
    fn validates_names_like_cpp() {
        assert!(valid_name("Іван"));
        assert!(valid_name("my voice_1-test"));
        assert!(valid_name(&"ї".repeat(40)));
        assert!(!valid_name(&"ї".repeat(41)));
        assert!(!valid_name(""));
        assert!(!valid_name("a/b"));
        assert!(!valid_name("a.b"));
        assert!(!valid_name("a\nb"));
        assert!(!valid_name("імʼя!"));
        assert!(!valid_name("a\n"));
    }

    #[test]
    fn reads_numpy_fixture() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/voice_npy_fixture.npy");
        let vector = load_npy_f32(&path).expect("load fixture");
        assert_eq!(vector.len(), VOICE_DIM);
        for (i, value) in vector.iter().enumerate() {
            let expected = i as f32 / 256.0 - 0.5;
            assert!((value - expected).abs() < 1e-7, "index {i}: {value}");
        }
    }

    #[test]
    fn roundtrips_and_rescans() {
        let dir = temp_dir("roundtrip");
        let voices = CustomVoices::new(&dir);
        assert!(voices.enabled());
        assert!(voices.names().is_empty());

        let mut embedding: Vec<f32> = (0..VOICE_DIM).map(|i| i as f32 / 128.0 - 1.0).collect();
        voices.save("Тест голос", &embedding).expect("save");
        assert!(voices.has("Тест голос"));
        assert_eq!(voices.get("Тест голос").unwrap(), embedding);
        assert_eq!(voices.names(), vec!["Тест голос".to_string()]);
        // 128-байтний заголовок + 256 float32 (як np.save).
        assert_eq!(
            fs::metadata(dir.join("Тест голос.npy")).unwrap().len(),
            128 + (VOICE_DIM * 4) as u64
        );

        embedding[0] = 42.0;
        voices.save("Тест голос", &embedding).expect("overwrite");
        assert_eq!(voices.get("Тест голос").unwrap()[0], 42.0);

        // Файл, створений повз API, підхоплюється ресканом.
        save_npy_f32(dir.join("External Voice.npy"), &[0.25; VOICE_DIM]).unwrap();
        assert_eq!(voices.get("External Voice").unwrap()[5], 0.25);

        // Пошкоджений .npy ігнорується.
        fs::write(dir.join("Broken.npy"), b"not a numpy file").unwrap();
        assert_eq!(voices.names().len(), 2);

        assert!(matches!(
            voices.save("bad/name", &embedding),
            Err(CustomVoiceError::InvalidName(_))
        ));
        assert!(matches!(
            voices.save("short", &[0.0; 100]),
            Err(CustomVoiceError::WrongDim(100))
        ));
        assert!(matches!(
            voices.get("нема"),
            Err(CustomVoiceError::Unknown(_))
        ));

        assert!(voices.remove("External Voice").unwrap());
        assert!(!voices.has("External Voice"));
        assert!(!voices.remove("External Voice").unwrap());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn disabled_storage_rejects_writes() {
        let voices = CustomVoices::new("");
        assert!(!voices.enabled());
        assert!(voices.names().is_empty());
        assert!(matches!(
            voices.save("x", &[0.0; VOICE_DIM]),
            Err(CustomVoiceError::Disabled)
        ));
        assert!(!voices.remove("x").unwrap());
    }
}
