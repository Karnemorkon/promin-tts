pub fn crate_name() -> &'static str {
    "speech-audio"
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(super::crate_name(), "speech-audio");
    }
}
