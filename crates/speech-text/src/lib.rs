pub fn crate_name() -> &'static str {
    "speech-text"
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(super::crate_name(), "speech-text");
    }
}
