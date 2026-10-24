//! Мінімальні UTF-8 утиліти (позиції наголосу — індекси codepoint, не байти).

/// Кількість байтів UTF-8-символу, що починається з `c` (1..4).
pub fn cp_len(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if c & 0xE0 == 0xC0 {
        2
    } else if c & 0xF0 == 0xE0 {
        3
    } else {
        4
    }
}

/// Кількість codepoint'ів.
pub fn len(s: &str) -> usize {
    s.chars().count()
}

/// Байтовий зсув до codepoint-індексу [0, len); `s.len()`, якщо поза межами.
pub fn byte_offset(s: &str, cp_index: usize) -> usize {
    let mut n = 0;
    let mut bytes = 0;
    for b in s.bytes() {
        if b & 0xC0 == 0x80 {
            bytes += 1;
            continue;
        }
        if n == cp_index {
            return bytes;
        }
        n += 1;
        bytes += 1;
    }
    if cp_index == n {
        bytes
    } else {
        s.len()
    }
}

/// Вставити `ins` ПЕРЕД codepoint-позицією `pos` (0-based). `pos >= len` → у кінець.
pub fn insert_at_cp(s: &str, pos: usize, ins: &str) -> String {
    let off = byte_offset(s, pos);
    let mut out = String::with_capacity(s.len() + ins.len());
    out.push_str(&s[..off]);
    out.push_str(ins);
    out.push_str(&s[off..]);
    out
}

/// UTF-8 encode codepoint.
pub fn encode_u32(cp: u32) -> String {
    char::from_u32(cp).map(String::from).unwrap_or_default()
}

/// Unicode lower для кодових точок, що зустрічаються в українському тексті
/// (кирилиця, латиниця Basic + Latin-1 Supplement). Парність з Python `.lower()`
/// для цього набору.
pub fn cp_lower(cp: u32) -> u32 {
    match cp {
        0x41..=0x5A => cp + 0x20,               // A-Z
        0xC0..=0xDE if cp != 0xD7 => cp + 0x20, // Latin-1 uppercase
        0x410..=0x42F => cp + 0x20,             // А-Я
        0x400..=0x40F => cp + 0x50,             // Ѐ-Џ
        0x490 => 0x491,                         // Ґ -> ґ
        _ => cp,
    }
}

/// Unicode upper для цього ж набору.
pub fn cp_upper(cp: u32) -> u32 {
    match cp {
        0x61..=0x7A => cp - 0x20,               // a-z
        0xE0..=0xFE if cp != 0xF7 => cp - 0x20, // Latin-1 lowercase
        0x430..=0x44F => cp - 0x20,             // а-я
        0x450..=0x45F => cp - 0x50,             // ѐ-џ
        0x491 => 0x490,                         // ґ -> Ґ
        _ => cp,
    }
}

/// Нижній регістр рядка (Unicode lower по codepoint).
pub fn unicode_lower(s: &str) -> String {
    s.chars()
        .map(|c| char::from_u32(cp_lower(c as u32)).unwrap_or(c))
        .collect()
}

/// Title-регістр: після не-літери — верхній, решта — нижній (як Python `.title()`).
pub fn unicode_title(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut boundary = true;
    for c in s.chars() {
        let cp = c as u32;
        let lc = cp_lower(cp);
        let uc = cp_upper(cp);
        let cased = lc != uc;
        let mapped = if cased && boundary {
            char::from_u32(uc).unwrap_or(c)
        } else {
            char::from_u32(lc).unwrap_or(c)
        };
        out.push(mapped);
        boundary = !cased;
    }
    out
}
