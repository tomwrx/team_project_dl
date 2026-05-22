# Privacy Masker — Team Project
## Dataset
pii_v2 dataset is publicly available on Google Drive:

[Download pii_v2 dataset](https://drive.google.com/drive/folders/1hGHE_ald1wqY_1g5CTNDwGaovNmWA5VD?usp=share_link)

### Dataset Structure
| Path | Description |
|------|-------------|
| `train.jsonl` | 3,639 screens |
| `val.jsonl` | 455 screens |
| `test.jsonl` | 455 screens |
| `images.zip` | All PNG files zipped |
| `images/` | Raw PNG files |

### Stats
- **Total screens:** 4,549
- **Split:** 80% train / 10% val / 10% test
- **Negative examples:** ~15% empty screens
- **Source:** rootsautomation/RICO-ScreenQA (HuggingFace)
- **Validator:** OK

### PII Labels
| Label | Phase |
|-------|-------|
| `email_address` | P1 |
| `phone_number` | P1 |
| `full_name` | P1 |
| `username` | P1 |
| `account_balance` | P1 |
| `transaction_amount` | P1 |
| `address` | P2 |
| `date_of_birth` | P2 |
| `other_sensitive` | P2 |

## Version history
| Version | Screens | Notes |
|---------|---------|-------|
| pii_v1 | 3,645 | train only, baseline KEYWORD_MAP |
| pii_v2 | 4,549 | added val+test splits (+25% screens) which were not included in first stream (69k out of 89k scanned), same baseline KEYWORD_MAP |
| pii_v3 | 10,645 | expanded KEYWORD_MAP keywords — NOT UPLOADED, too many false positives |
| pii_v4 | {len(final_entries)} | data-driven KEYWORD_MAP, removed keywords with >40% non-PII rate, same labels with updated KEYWORD_MAP |
