# NT-SSM dataset conversion

`lastfm/` and `ml-1m/` are converted from the released NT-SSM splits for
cross-dataset NR-GCF experiments.

## Protocol

- Source `train.txt` and `valid.txt` are merged into NR-GCF `train.txt`.
- The converted test split retains only interactions whose user and item both
  occur in the merged training graph.  Retained interactions keep source order.
- Released numeric user and item IDs are preserved without remapping.  The IDs
  are globally zero-based and contiguous in both datasets.
- Source files contain one `user item 1` triple per line.  NR-GCF files contain
  one user followed by all of that user's items per line.
- There are no duplicate interactions and no train/valid/test overlap in either
  released dataset.

The deterministic NR-GCF training-edge order is grouped by user.  Within each
user, source training items keep their order and source validation items are
appended in their original order.

## Test cold-start filtering

Cold-start interactions are removed only after train and validation have been
merged.  The source files remain untouched; the converted NR-GCF test files use
the following training-closed protocol:

| Dataset | Merged train | Source test | Filtered cold edges | Converted test | Test users removed entirely |
|---|---:|---:|---:|---:|---:|
| LastFM | 73,458 | 18,321 | 2,376 | 15,945 | 4 |
| MovieLens-1M | 671,630 | 164,848 | 29 | 164,819 | 0 |

LastFM filtering removes 2,247 training-unseen item IDs; MovieLens-1M removes 26.
Neither dataset has a training-unseen test user, but four LastFM users lose all
test interactions because all of their test items are cold.  Those users are not
included in NR-GCF's Recall/NDCG denominator.

Each dataset's `conversion_metadata.json` records source and converted SHA-256
hashes, source/retained counts, overlap checks, ID checks, and the complete
cold-start filtering audit.

## Reproduce the conversion

From the NR-GCF repository root:

```bash
python3 code/convert_ntssm_datasets.py \
  --source-root /path/to/NT-SSM/dataset \
  --output-root data
```

The converter rejects malformed triples, duplicate pairs, split overlap, and
non-contiguous global IDs rather than choosing a silent repair policy.
