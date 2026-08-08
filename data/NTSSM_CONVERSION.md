# NT-SSM dataset conversion

`lastfm/` and `ml-1m/` are converted from the released NT-SSM splits for
cross-dataset NR-GCF experiments.

## Protocol

- Source `train.txt` is the only NR-GCF training split.  Source `valid.txt` is
  excluded completely and is not merged or evaluated.
- The converted test split retains only interactions whose user and item both
  occur in source train.  Retained interactions keep source order.
- Released numeric user and item IDs are preserved without remapping.  The IDs
  are globally zero-based and contiguous in both datasets.
- Source files contain one `user item 1` triple per line.  NR-GCF files contain
  one user followed by all of that user's items per line.
- There are no duplicate interactions and no train/valid/test overlap in either
  released dataset.

The deterministic NR-GCF training-edge order is the released source train order
grouped by user.

## No-cold-start evaluation protocol

Cold-start status is computed against source train only.  The committed NR-GCF
test files remove cold interactions without moving them into training:

| Dataset | Train | Valid excluded | Source test | Removed cold edges | Converted test |
|---|---:|---:|---:|---:|---:|
| LastFM | 64,315 | 9,143 | 18,321 | 2,543 | 15,778 |
| MovieLens-1M | 590,733 | 80,897 | 164,848 | 34 | 164,814 |

Neither dataset contains a training-unseen test user.  LastFM removes 2,382
training-unseen item IDs and MovieLens-1M removes 31.  Five LastFM users lose all
test interactions because every one of their test items is cold; they therefore
do not enter the Recall/NDCG denominator.  Validation and test interactions are
never added to train.

Each dataset's `conversion_metadata.json` records source and converted SHA-256
hashes, source/converted counts, overlap checks, ID checks, and the cold-start
audit.

## Reproduce the conversion

From the NR-GCF repository root:

```bash
python3 code/convert_ntssm_datasets.py \
  --source-root /path/to/NT-SSM/dataset \
  --output-root data
```

For a separate diagnostic conversion that deliberately retains cold-start test
interactions, add `--retain-cold-start`.  This option is not used by the
committed datasets.

To reproduce the previous train+valid protocol for diagnostics, add
`--merge-validation`.  This option is also not used by the committed datasets.

The converter rejects malformed triples, duplicate pairs, split overlap, and
non-contiguous global IDs rather than choosing a silent repair policy.
