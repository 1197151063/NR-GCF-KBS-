# NT-SSM dataset conversion

`lastfm/` and `ml-1m/` are converted from the released NT-SSM splits for
cross-dataset NR-GCF experiments.

## Protocol

- Source `train.txt` and `valid.txt` are merged into NR-GCF `train.txt`.
- Source `test.txt` remains the test split.  No test interaction is removed,
  relocated, or used for training.
- Released numeric user and item IDs are preserved without remapping.  The IDs
  are globally zero-based and contiguous in both datasets.
- Source files contain one `user item 1` triple per line.  NR-GCF files contain
  one user followed by all of that user's items per line.
- There are no duplicate interactions and no train/valid/test overlap in either
  released dataset.

The deterministic NR-GCF training-edge order is grouped by user.  Within each
user, source training items keep their order and source validation items are
appended in their original order.

## Important test cold-start property

Keeping the test split unchanged leaves test-only items:

| Dataset | Merged train edges | Test edges | Test-only items | Affected test edges |
|---|---:|---:|---:|---:|
| LastFM | 73,458 | 18,321 | 2,247 | 2,376 |
| MovieLens-1M | 671,630 | 164,848 | 26 | 29 |

NR-GCF allocates embeddings for these globally valid item IDs, but they have no
training-edge propagation.  They are intentionally retained because the chosen
protocol requires an unchanged test split.  This differs from loaders that
silently skip entities unseen in training and should be disclosed when comparing
numbers across codebases.

Each dataset's `conversion_metadata.json` records source and converted SHA-256
hashes, counts, overlap checks, ID checks, and cold-start counts.

## Reproduce the conversion

From the NR-GCF repository root:

```bash
python3 code/convert_ntssm_datasets.py \
  --source-root /path/to/NT-SSM/dataset \
  --output-root data
```

The converter rejects malformed triples, duplicate pairs, split overlap, and
non-contiguous global IDs rather than choosing a silent repair policy.
