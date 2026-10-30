//! CTC greedy decode: argmax по кадрах, згортання повторів і blank.

use anyhow::{bail, Result};

/// CTC greedy decode без словника.
///
/// `logits` — неперервний тензор `[frames, vocab]` (batch = 1).
/// Повтори того самого токена згортаються, але токен після blank
/// вважається новим (стандартна поведінка CTC).
pub fn greedy_ctc(logits: &[f32], frames: usize, vocab: usize, blank: u32) -> Result<Vec<u32>> {
    if frames == 0 || vocab == 0 {
        return Ok(Vec::new());
    }
    let expected = frames
        .checked_mul(vocab)
        .ok_or_else(|| anyhow::anyhow!("CTC tensor size overflow"))?;
    if logits.len() != expected {
        bail!(
            "CTC tensor has {} values, expected {frames}x{vocab}={expected}",
            logits.len()
        );
    }

    let mut tokens = Vec::new();
    let mut previous = None;
    for frame in 0..frames {
        let row = &logits[frame * vocab..(frame + 1) * vocab];
        let mut best = 0usize;
        for (index, value) in row.iter().enumerate().skip(1) {
            if value.total_cmp(&row[best]).is_gt() {
                best = index;
            }
        }
        let token = best as u32;
        if token != blank && Some(token) != previous {
            tokens.push(token);
        }
        previous = Some(token);
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_hot(frames: &[usize], vocab: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; frames.len() * vocab];
        for (frame, &token) in frames.iter().enumerate() {
            logits[frame * vocab + token] = 1.0;
        }
        logits
    }

    #[test]
    fn collapses_repeats_and_drops_blank() {
        let logits = one_hot(&[0, 2, 2, 0, 2, 3, 3, 0, 0, 4], 5);
        let tokens = greedy_ctc(&logits, 10, 5, 0).unwrap();
        assert_eq!(tokens, vec![2, 2, 3, 4]);
    }

    #[test]
    fn blank_separates_equal_tokens() {
        let logits = one_hot(&[2, 0, 2], 5);
        assert_eq!(greedy_ctc(&logits, 3, 5, 0).unwrap(), vec![2, 2]);
    }

    #[test]
    fn empty_or_invalid_tensor() {
        assert!(greedy_ctc(&[], 0, 5, 0).unwrap().is_empty());
        assert!(greedy_ctc(&[0.0; 3], 2, 5, 0).is_err());
    }

    #[test]
    fn all_blank_produces_nothing() {
        let logits = one_hot(&[0, 0, 0], 4);
        assert!(greedy_ctc(&logits, 3, 4, 0).unwrap().is_empty());
    }
}
