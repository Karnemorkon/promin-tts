//! Інтерфейс нейромережевого вербалізатора.
//!
//! Реалізація живе в `speech-inference` (ONNX Runtime), щоб `speech-text`
//! не залежав від ORT. Сервіс зберігає `Option<Arc<dyn NeuralVerbalizer>>`
//! і використовує rule-based [`crate::verbalize::verbalize`] як fallback.

/// Нейромережева вербалізація тексту з цифрами.
pub trait NeuralVerbalizer: Send + Sync {
    /// Повертає текст із розгорнутими числами/датами/часом.
    ///
    /// Реалізація може повернути оригінальний текст для речень, які не
    /// вдалося обробити (як у C++ `GemmaVerbalizer`).
    fn verbalize(&self, text: &str) -> anyhow::Result<String>;
}
