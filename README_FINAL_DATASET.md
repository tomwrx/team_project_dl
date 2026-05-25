# Privacy Masker — Team Project

A deep learning pipeline for high-precision PII (Personally Identifiable Information) detection and masking in mobile UI screenshots. Built on the [RICO-ScreenQA](https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA) dataset.

---

## Dataset — `pii_v5`

📁 [Download pii_v5 on Google Drive](https://drive.google.com/drive/folders/1DnbkCIIex6pEyrxMQenOcWaxxL8l7PW-?usp=share_link)

### Structure

```
pii_v5/
├── train.jsonl       # 6,999 PII screens + 913 negative examples
├── val.jsonl         #   875 PII screens + 114 negative examples
├── test.jsonl        #   882 PII screens + 115 negative examples
└── images.zip        # All UI screenshots (unzip to images/)
```

Each `.jsonl` line follows this schema:

```json
{
  "image":        "images/<screen_id>.png",
  "image_width":  1080,
  "image_height": 1920,
  "screen_id":    "12345",
  "objects": [
    {
      "bbox":            [x1, y1, x2, y2],
      "label":           "email_address",
      "source_question": "What is the support email address?",
      "source_answer":   "support@example.com"
    }
  ]
}
```

Screens with no PII have `"objects": []` and serve as hard negative examples (~12–13% of each split).

---

### Label Distribution

| Label | Train | Val | Test | Priority |
|-------|------:|----:|-----:|:--------:|
| `address` | 3,135 | 373 | 417 | P1 |
| `transaction_amount` | 2,316 | 323 | 279 | P2 |
| `email_address` | 1,975 | 240 | 229 | P1 |
| `other_sensitive` | 1,260 | 183 | 138 | P2 |
| `date_of_birth` | 884 | 115 | 121 | P1 |
| `phone_number` | 816 | 109 | 92 | P1 |
| `username` | 570 | 66 | 76 | P2 |
| `full_name` | 201 | 27 | 25 | P1 |
| `account_balance` | 153 | 18 | 19 | P2 |
| **Total screens** | **7,912** | **989** | **997** | |

**P1** — core PII, highest detection priority (identity + contact).  
**P2** — extended PII, financial and credential data.

---

### PII Categories & Regex Rules 

All questions are classified using compiled regex patterns matched against the question text.

> **Note:** The keywords listed below represent the *primary signals* identified during n-gram analysis, but the regex patterns intentionally match a wider surface — e.g. `email` also catches phrasing like *"enter your mail"*, and `balance` catches *"current account balance"* — so actual dataset coverage is broader than the keyword list implies.

| Label | Keywords / Patterns | Regex |
|-------|--------------------|----|
| `email_address` | email, mail, gmail, email address, email id | `\b(?:email\|mail\|gmail)(?:\s+(?:address\|id))?\b` |
| `phone_number` | phone, contact, helpline, phone number | `\b(?:phone\|contact\|helpline)(?:\s+number)?\b` |
| `full_name` | first name, last name, full name, display name | `\b(?:first\|last\|full\|display)\s+name\b` |
| `address` | address, street, zip, postal, location, city, country, destination | `\b(?:address\|street\|zip\|postal\|location\|city\|country\|destination)\b` |
| `date_of_birth` | date of birth, dob, birthday, born, age | `\b(?:date\s+of\s+birth\|dob\|birthday\|born\|age)\b` |
| `account_balance` | balance, account balance | `\b(?:account\s+)?balance\b` |
| `transaction_amount` | charge, price, cost, fee, rent; transaction/payment/total amount | `\b(?:charge\|price\|cost\|fee\|rent)\b\|\b(?:transaction\|payment\|total\|purchase\|monthly)\s+(?:amount\|price\|cost\|payment)\b` |
| `username` | username, login name, user id | `\b(?:username\|login\s+name\|user\s+id)\b` |
| `other_sensitive` | gender, password, passcode, pin, weight, height | `\b(?:gender\|password\|passcode\|pin\|weight\|height)\b` |

---

## Version History

| Version | Screens | Notes |
|---------|--------:|-------|
| `pii_v1` | 3,645 | Baseline — keyword matching only, no val/test split. |
| `pii_v2` | 4,549 | Added validation and test splits. |
| `pii_v5` | **9,989** | Data-driven regex engine, stratified splitting, pruned labels, biometrics added. |

### V5 Upgrades Over Previous Versions

**1. Stratified Multi-Label Splitting — "Rarest Label Heuristic"**  
Each screen is assigned a *primary label* equal to the globally rarest PII label it contains. Screens are then grouped by primary label and split 80/10/10. This ensures rare categories (`full_name`, `account_balance`) are proportionally represented in val and test instead of being absent.

**2. Precision-First Regex Engine (V6)**  
Replaced broad keyword lists with compiled `REGEX_KEYWORD_MAP` patterns. High-noise keywords such as bare `"name"` and `"date"` (which caused >40% false positives in earlier versions) were removed. Only structurally unambiguous terms and bigrams are retained.

**3. Biometric Compliance**  
`weight` and `height` added to the `other_sensitive` label for fitness and health app coverage.

**4. Label Pruning**  
`account_number` and `id_number` labels deprecated — both had low confidence due to the absence of reliable structural patterns in the source QA data, leading to noisy bounding box annotations.

---

### Key Regex Changes — V6 Engine

| Label | Change | Rationale |
|-------|--------|-----------|
| `full_name` | Removed bare `"name"` keyword | Too ambiguous — matched product names, field labels, etc. Kept only `first/last/full/display name` bigrams. |
| `date_of_birth` | Strict patterns only | `"date"` alone matched app version dates, news article dates, etc. Retained only `dob`, `birthday`, `born`, `age`. |
| `account_balance` | Added contextual bigrams | `"account balance"` pattern reduces collision with non-financial uses of `"balance"`. |
| `other_sensitive` | Expanded to biometrics | Added `weight`, `height` alongside `gender`, `password`, `passcode`, `pin`. |
| `account_number` | **Deprecated** | No reliable regex signature in QA question phrasing; high false-positive rate. |
| `id_number` | **Deprecated** | Same issue — `"id"` is used generically throughout app UI text. |

---

## Negative Examples

~12–13% of each split consists of screens with `"objects": []`. These are UI screenshots where no question matched any PII regex pattern. They serve as hard negatives to reduce the model's tendency to predict PII on every screen.

```
Train: 913 negative / 7,912 total = 11.5%
Val:   114 negative /   989 total = 11.5%
Test:  115 negative /   997 total = 11.5%
```
