use std::sync::LazyLock;

use regex::{Captures, Regex};

use crate::sanitizer::de_anonymizer::{DeAnonymizer, PiiKind};

static PII_COMBINED_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?P<e164_phone>\+\d{10,15}\b)|",
        r"(?P<card>\b\d{4}(?:[ -]\d{4}){2}[ -]\d{1,4}\b|\b\d{4}(?:[ -]\d{4}){3}(?:[ -]\d{1,3})?\b|\b\d{4}[ -]\d{6}[ -]\d{5}\b|\b\d{13,19}\b)|",
        r"(?P<cpf_fmt>\b\d{3}\.\d{3}\.\d{3}-\d{2}\b)|",
        r"(?P<cpf_ctx>(?i)\b(?P<cpf_ctx_prefix>cpf\s*[:#-]?\s*)(?P<cpf_ctx_digits>\d{11})\b)|",
        r"(?P<cpf_raw>\b\d{11}\b)|",
        r"(?P<email>\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b)|",
        r"(?P<phone>(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,3}\)[\s.-]?|\b\d{2,4}[\s.-])\d{4,5}[\s.-]?\d{4}\b)",
    ))
    .expect("combined PII regex must compile")
});

/// Egress Data Loss Prevention (DLP) masker for Brazilian CPFs, Luhn-valid credit card PANs,
/// email addresses, and phone numbers.
pub struct PiiMasker;

impl PiiMasker {
    /// Mask all detected PII in `input`, registering reversible tokens in `de_anonymizer`.
    pub fn mask(input: &str, de_anonymizer: &mut DeAnonymizer) -> String {
        PII_COMBINED_RE
            .replace_all(input, |caps: &Captures<'_>| {
                if let Some(m) = caps.name("e164_phone") {
                    de_anonymizer.register(PiiKind::Phone, m.as_str())
                } else if let Some(m) = caps.name("card") {
                    let matched = m.as_str();
                    if luhn_valid(matched) {
                        de_anonymizer.register(PiiKind::CreditCard, matched)
                    } else {
                        matched.to_string()
                    }
                } else if let Some(m) = caps.name("cpf_fmt") {
                    de_anonymizer.register(PiiKind::Cpf, m.as_str())
                } else if caps.name("cpf_ctx").is_some() {
                    let prefix = caps.name("cpf_ctx_prefix").map_or("", |m| m.as_str());
                    let digits = caps.name("cpf_ctx_digits").map_or("", |m| m.as_str());
                    let token = de_anonymizer.register(PiiKind::Cpf, digits);
                    format!("{prefix}{token}")
                } else if let Some(m) = caps.name("cpf_raw") {
                    let matched = m.as_str();
                    if is_valid_cpf(matched) {
                        de_anonymizer.register(PiiKind::Cpf, matched)
                    } else {
                        matched.to_string()
                    }
                } else if let Some(m) = caps.name("email") {
                    de_anonymizer.register(PiiKind::Email, m.as_str())
                } else if let Some(m) = caps.name("phone") {
                    de_anonymizer.register(PiiKind::Phone, m.as_str())
                } else {
                    caps[0].to_string()
                }
            })
            .into_owned()
    }
}

/// Validate a credit card PAN (13–19 digits, optionally separated by spaces or hyphens)
/// using the Luhn mod-10 checksum algorithm without heap allocation.
pub fn luhn_valid(candidate: &str) -> bool {
    let mut buf = [0u8; 20];
    let mut len = 0usize;
    for b in candidate.bytes() {
        if b.is_ascii_digit() {
            if len >= 19 {
                return false;
            }
            buf[len] = b - b'0';
            len += 1;
        } else if b != b' ' && b != b'-' {
            return false;
        }
    }
    if !(13..=19).contains(&len) {
        return false;
    }

    let mut sum: u32 = 0;
    let mut double = false;
    for &digit in buf[..len].iter().rev() {
        let mut d = u32::from(digit);
        if double {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        sum += d;
        double = !double;
    }

    sum.is_multiple_of(10)
}

/// Validate an 11-digit Brazilian CPF using its two mod-11 check digits.
pub fn is_valid_cpf(digits: &str) -> bool {
    if digits.len() != 11 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }

    let bytes = digits.as_bytes();
    // Reject known invalid all-identical sequences (e.g., 00000000000, 11111111111).
    if bytes.iter().all(|&b| b == bytes[0]) {
        return false;
    }

    let mut nums = [0u32; 11];
    for (i, &b) in bytes.iter().enumerate() {
        nums[i] = u32::from(b - b'0');
    }

    let sum1: u32 = (0..9).map(|i| nums[i] * (10 - i as u32)).sum();
    let rem1 = (sum1 * 10) % 11;
    let check1 = if rem1 >= 10 { 0 } else { rem1 };
    if check1 != nums[9] {
        return false;
    }

    let sum2: u32 = (0..10).map(|i| nums[i] * (11 - i as u32)).sum();
    let rem2 = (sum2 * 10) % 11;
    let check2 = if rem2 >= 10 { 0 } else { rem2 };
    check2 == nums[10]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_formatted_and_unformatted_brazilian_cpf() {
        let mut deanon = DeAnonymizer::new();
        let input = "Customer CPF 123.456.789-09 and alt 52998224725 and context CPF: 00000000191";
        let masked = PiiMasker::mask(input, &mut deanon);

        assert!(!masked.contains("123.456.789-09"));
        assert!(!masked.contains("52998224725"));
        assert!(!masked.contains("00000000191"));
        assert_eq!(deanon.len(), 3);
        assert_eq!(deanon.restore(&masked), input);
    }

    #[test]
    fn masks_luhn_valid_pan_and_skips_non_luhn_order_ids() {
        let mut deanon = DeAnonymizer::new();
        let input = "Card 4532-0151-1283-0366 charged twice for order 1234567890123";
        let masked = PiiMasker::mask(input, &mut deanon);

        assert!(!masked.contains("4532-0151-1283-0366"));
        assert!(
            masked.contains("1234567890123"),
            "non-Luhn order ID must not be masked"
        );
        assert_eq!(deanon.len(), 1);
        assert_eq!(deanon.restore(&masked), input);
    }

    #[test]
    fn masks_email_and_phone_and_restores_both() {
        let mut deanon = DeAnonymizer::new();
        let input =
            "Contact ana.silva@example.com.br or +55 (11) 98765-4321 regarding Order #12345.";
        let masked = PiiMasker::mask(input, &mut deanon);

        assert!(!masked.contains("ana.silva@example.com.br"));
        assert!(!masked.contains("+55 (11) 98765-4321"));
        assert!(masked.contains("Order #12345"));
        assert_eq!(deanon.len(), 2);
        assert_eq!(deanon.restore(&masked), input);
    }

    #[test]
    fn clean_prompt_with_order_id_is_unchanged() {
        let mut deanon = DeAnonymizer::new();
        let input = "Order #12345 is delayed by 3 days.";
        let masked = PiiMasker::mask(input, &mut deanon);

        assert_eq!(masked, input);
        assert!(deanon.is_empty());
    }

    #[test]
    fn luhn_valid_enforces_length_and_checksum() {
        assert!(luhn_valid("4532015112830366"));
        assert!(luhn_valid("4532-0151-1283-0366"));
        assert!(!luhn_valid("1234567890123"));
        assert!(!luhn_valid("424242424242")); // 12 digits (too short)
        assert!(!luhn_valid("45320151128303661234")); // 20 digits (oversized)
        assert!(!luhn_valid("4532-0151-1283-036a")); // non-digit
    }

    #[test]
    fn invalid_unformatted_cpfs_remain_unmasked() {
        let mut deanon = DeAnonymizer::new();
        let input = "Tracking codes 11111111111 and 12345678900 are not valid CPFs.";
        let masked = PiiMasker::mask(input, &mut deanon);

        assert_eq!(masked, input);
        assert!(deanon.is_empty());
    }

    #[test]
    fn masks_twelve_digit_and_e164_phones_as_phone() {
        let mut deanon = DeAnonymizer::new();
        let input = "Call 1234-5678-9012 or +5511987654321 for support.";
        let masked = PiiMasker::mask(input, &mut deanon);

        assert_eq!(
            masked,
            "Call <<PII:PHONE_1>> or <<PII:PHONE_2>> for support."
        );
        assert_eq!(deanon.restore(&masked), input);
    }
}
