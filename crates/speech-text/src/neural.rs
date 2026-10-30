//! Інтерфейси нейромережевих компонентів (вербалізатор, accentor).
//!
//! Реалізації живуть у `speech-inference` (ONNX Runtime), щоб `speech-text`
//! не залежав від ORT. Сервіс зберігає `Option<Arc<dyn ...>>` і використовує
//! rule-based fallback.

/// Нейромережева вербалізація тексту з цифрами.
pub trait NeuralVerbalizer: Send + Sync {
    /// Повертає текст із розгорнутими числами/датами/часом.
    ///
    /// Реалізація може повернути оригінальний текст для речень, які не
    /// вдалося обробити (як у C++ `GemmaVerbalizer`).
    fn verbalize(&self, text: &str) -> anyhow::Result<String>;
}

/// Нейромережевий наголос для слів, яких немає у словнику (accentor-lite).
pub trait NeuralStress: Send + Sync {
    /// Повертає слово з U+0301 після наголошеної літери або без змін.
    fn stress(&self, word: &str) -> anyhow::Result<String>;
}
