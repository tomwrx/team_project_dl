# Privacy Masker — Team Project

## Dataset
`pii_v5` dataset is prepared for high-precision PII detection and masking.

[Download pii_v5 dataset](https://drive.google.com/drive/folders/1DnbkCIIex6pEyrxMQenOcWaxxL8l7PW-?usp=share_link)

### Dataset Structure
| Path | Description |
|------|-------------|
| `train.jsonl` | 6,999 PII screens + 913 negative |
| `val.jsonl` | 875 PII screens + 114 negative |
| `test.jsonl` | 882 PII screens + 115 negative |
| `images.zip` | Compressed archive of all UI screenshots |

### V5 Label Distribution
| Label | Train | Val | Test | Priority |
|-------|-------|-----|------|----------|
| `address` | 3135 | 373 | 417 | P1 |
| `transaction_amount` | 2316 | 323 | 279 | P2 |
| `email_address` | 1975 | 240 | 229 | P1 |
| `other_sensitive` | 1260 | 183 | 138 | P2 |
| `date_of_birth` | 884 | 115 | 121 | P1 |
| `phone_number` | 816 | 109 | 92 | P1 |
| `username` | 570 | 66 | 76 | P2 |
| `full_name` | 201 | 27 | 25 | P1 |
| `account_balance` | 153 | 18 | 19 | P2 |

### V5 Technical Upgrades
1. **Stratified Multi-Label Splitting:** Implemented "Rarest Label Heuristic" to ensure rare PII labels (e.g., biometrics) are evenly distributed across Train/Val/Test splits.
2. **Precision-First Regex Engine:** Upgraded to `REGEX_KEYWORD_MAP`. Eliminated high-noise keywords (e.g., 'name', 'date') that caused >40% false positives previous datasets.
3. **Biometric Compliance:** Added `weight` and `height` detection under `other_sensitive` label for enhanced fitness/health app coverage.
4. **Label Pruning:** Deprecated `account_number` and `id_number` labels due to low confidence and lack of structural patterns in the source data.

## Version history
| Version | Screens | Notes |
|---------|---------|-------|
| `pii_v1` | 3,645 | Baseline; keywords only. |
| `pii_v2` | 4,549 | Added validation/test splits. |
| `pii_v5` | 9,989 | Data-driven Regex implementation. Stratified splitting, pruned labels, biometrics added. |

## Key Regex Changes (V6 Engine)
| Label | Status | Logic |
|-------|--------|-------|
| `full_name` | Refined | Removed "name" (too ambiguous); kept specific titles/patterns. |
| `date_of_birth` | Refined | Strict patterns only (DOB, birthday, age). |
| `account_balance` | Added | Contextual bigrams (e.g., "account balance", "money amount"). |
| `other_sensitive`| Expanded | Includes `weight`, `height`, `gender`, `pin`, `passcode`. |
