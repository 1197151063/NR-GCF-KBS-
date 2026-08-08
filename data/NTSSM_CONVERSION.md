# NT-SSM dataset conversion

`lastfm/` and `ml-1m/` are converted from the released NT-SSM splits for
cross-dataset NR-GCF experiments.

## Protocol

- Source `train.txt` and `valid.txt` are merged into NR-GCF `train.txt`.
- The converted test split preserves every released source test interaction and
  its order.  Cold-start interactions are retained by default.
- Released numeric user and item IDs are preserved without remapping.  The IDs
  are globally zero-based and contiguous in both datasets.
- Source files contain one `user item 1` triple per line.  NR-GCF files contain
  one user followed by all of that user's items per line.
- There are no duplicate interactions and no train/valid/test overlap in either
  released dataset.

The deterministic NR-GCF training-edge order is grouped by user.  Within each
user, source training items keep their order and source validation items are
appended in their original order.

## Test cold-start audit

Cold-start status is computed only after train and validation have been merged.
The committed NR-GCF test files retain these interactions:

| Dataset | Merged train | Source/converted test | Retained cold edges | Training-unseen test items |
|---|---:|---:|---:|---:|
| LastFM | 73,458 | 18,321 | 2,376 | 2,247 |
| MovieLens-1M | 671,630 | 164,848 | 29 | 26 |

Neither dataset contains a training-unseen test user.  NR-GCF allocates the
globally valid test item IDs, but cold items receive no training-edge propagation.
The metadata makes this protocol explicit so filtered and unfiltered results are
not mixed.

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

For a separate training-closed diagnostic conversion, add
`--filter-cold-start`.  This option is not used by the committed datasets.

The converter rejects malformed triples, duplicate pairs, split overlap, and
non-contiguous global IDs rather than choosing a silent repair policy.
