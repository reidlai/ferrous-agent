use std::collections::HashMap;
use std::sync::LazyLock;

use regex::{Captures, Regex};

static TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<<PII:(?:CPF|CARD|EMAIL|PHONE)_\d+>>").expect("token regex must compile")
});

/// Categories of PII pseudonymized by [`crate::sanitizer::PiiMasker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PiiKind {
    /// Brazilian Cadastro de Pessoas Físicas (CPF) number.
    Cpf,
    /// Luhn-valid payment card Primary Account Number (PAN).
    CreditCard,
    /// Email address.
    Email,
    /// Telephone number.
    Phone,
}

impl PiiKind {
    /// Returns the deterministic placeholder prefix used in tokens (`<<PII:<PREFIX>_N>>`).
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Cpf => "CPF",
            Self::CreditCard => "CARD",
            Self::Email => "EMAIL",
            Self::Phone => "PHONE",
        }
    }
}

/// Session-scoped reversible token mapper for pseudonymized PII.
///
/// Maps original PII values to deterministic placeholders (`<<PII:CPF_1>>`,
/// `<<PII:CARD_1>>`, `<<PII:EMAIL_1>>`, `<<PII:PHONE_1>>`) and restores them locally
/// via [`DeAnonymizer::restore`].
#[derive(Debug, Clone, Default)]
pub struct DeAnonymizer {
    token_to_original: HashMap<String, String>,
    original_to_token: HashMap<(PiiKind, String), String>,
    counters: HashMap<PiiKind, usize>,
}

impl DeAnonymizer {
    /// Create an empty session-scoped [`DeAnonymizer`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an original PII string and return its deterministic placeholder token.
    ///
    /// Repeated registrations of the same `(kind, original)` pair within the same
    /// session return the existing token.
    pub fn register(&mut self, kind: PiiKind, original: &str) -> String {
        let key = (kind, original.to_string());
        if let Some(existing) = self.original_to_token.get(&key) {
            return existing.clone();
        }

        let counter = self.counters.entry(kind).or_insert(0);
        *counter += 1;
        let token = format!("<<PII:{}_{}>>", kind.prefix(), *counter);

        self.token_to_original
            .insert(token.clone(), original.to_string());
        self.original_to_token.insert(key, token.clone());
        token
    }

    /// Restore all registered placeholder tokens in `text` back to their original PII values
    /// in a single pass, preventing double-replacement if an original value contains a token.
    pub fn restore(&self, text: &str) -> String {
        if self.token_to_original.is_empty() {
            return text.to_string();
        }

        TOKEN_RE
            .replace_all(text, |caps: &Captures<'_>| {
                let token = &caps[0];
                self.token_to_original
                    .get(token)
                    .map_or(token, String::as_str)
                    .to_string()
            })
            .into_owned()
    }

    /// Number of distinct PII values tracked in this session.
    pub fn len(&self) -> usize {
        self.token_to_original.len()
    }

    /// Returns `true` if no PII values have been registered.
    pub fn is_empty(&self) -> bool {
        self.token_to_original.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_deterministic_tokens_and_restores_round_trip() {
        let mut deanon = DeAnonymizer::new();
        assert!(deanon.is_empty());

        let cpf_tok = deanon.register(PiiKind::Cpf, "123.456.789-09");
        let cpf_tok_dup = deanon.register(PiiKind::Cpf, "123.456.789-09");
        let email_tok = deanon.register(PiiKind::Email, "ana.silva@example.com.br");
        let phone_tok = deanon.register(PiiKind::Phone, "+55 (11) 98765-4321");
        let card_tok = deanon.register(PiiKind::CreditCard, "4532-0151-1283-0366");

        assert_eq!(cpf_tok, "<<PII:CPF_1>>");
        assert_eq!(cpf_tok_dup, "<<PII:CPF_1>>");
        assert_eq!(email_tok, "<<PII:EMAIL_1>>");
        assert_eq!(phone_tok, "<<PII:PHONE_1>>");
        assert_eq!(card_tok, "<<PII:CARD_1>>");
        assert_eq!(deanon.len(), 4);

        let masked = format!("Customer {cpf_tok} ({email_tok}, {phone_tok}) paid with {card_tok}.");
        let restored = deanon.restore(&masked);
        assert_eq!(
            restored,
            "Customer 123.456.789-09 (ana.silva@example.com.br, +55 (11) 98765-4321) paid with 4532-0151-1283-0366."
        );
    }

    #[test]
    fn restore_does_not_double_replace_when_original_contains_token() {
        let mut deanon = DeAnonymizer::new();
        let cpf_tok = deanon.register(PiiKind::Cpf, "123.456.789-09");
        let email_tok = deanon.register(PiiKind::Email, "user+<<PII:CPF_1>>@example.com");

        let masked = format!("{email_tok} and {cpf_tok}");
        let restored = deanon.restore(&masked);
        assert_eq!(
            restored,
            "user+<<PII:CPF_1>>@example.com and 123.456.789-09"
        );
    }
}
