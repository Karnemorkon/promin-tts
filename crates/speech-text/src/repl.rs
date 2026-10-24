//! Перетворення шаблону заміни у синтаксис fancy-regex.
//!
//! C++/Python-референс використовує `$1`, `$2`, ...; fancy-regex трактує
//! `$1i` як групу з іменем `1i` (а не групу 1 + літеру `i`), тому всі
//! цифрові посилання переписуємо у безпечну форму `${1}`.

/// `$12abc` -> `${12}abc`, `$$` лишається як є (лише `$` + цифри).
pub(crate) fn to_fancy_template(template: &str) -> String {
    let mut out = String::with_capacity(template.len() + 8);
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '$' {
            let mut digits = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_digit() {
                    digits.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            if digits.is_empty() {
                out.push('$');
            } else {
                out.push_str("${");
                out.push_str(&digits);
                out.push('}');
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refs_are_braced() {
        assert_eq!(to_fancy_template("$1i"), "${1}i");
        assert_eq!(to_fancy_template("$12x$3"), "${12}x${3}");
        assert_eq!(to_fancy_template("$1$2"), "${1}${2}");
    }

    #[test]
    fn plain_dollar_untouched() {
        assert_eq!(to_fancy_template("a$b"), "a$b");
        assert_eq!(to_fancy_template(""), "");
    }
}
