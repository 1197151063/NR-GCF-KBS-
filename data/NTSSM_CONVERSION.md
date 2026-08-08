# LastFM and MovieLens dataset preparation

The two datasets intentionally use different source protocols.

## LastFM

- Source: NT-SSM `dataset/lastfm/{train,valid,test}.txt` triples.
- Train: source train and valid are merged.
- Test: remains the test split, then interactions with a merged-train-unseen
  user or item are filtered.
- IDs are preserved without remapping.

LastFM has 73,458 merged training interactions.  Its source test has 18,321
interactions; 2,376 cold interactions are filtered, leaving 15,945.

## MovieLens

- Source: the user-supplied `/Users/chenyijun/Desktop/KBS2026/ml_data` grouped
  `train.txt` and `test.txt`, not the old NT-SSM ML-1M triples.
- Train/test are already in NR-GCF grouped format and are imported without ID
  remapping.
- No validation split exists in this supplied version.
- The supplied test already has no cold-start interaction, so filtering removes
  zero edges.

MovieLens contains 6,022 users, 3,043 items, 796,244 train interactions, and
99,455 test interactions.  Train and test have no duplicates or overlap.

## Shared guarantees

- Test interactions are never moved into train.
- Converted test is training-closed.
- Edge order is deterministic.
- Each `conversion_metadata.json` records source/converted SHA-256 hashes,
  counts, overlap checks, ID checks, cold-start audit, and the dataset-specific
  protocol.

## Reproduce the conversion

From the NR-GCF repository root:

```bash
python3 code/convert_ntssm_datasets.py \
  --source-root /path/to/NT-SSM/dataset \
  --movielens-source-root /path/to/ml_data \
  --output-root data
```

For a separate diagnostic conversion that deliberately retains cold-start test
interactions, add `--retain-cold-start`.  This option is not used by the
committed datasets.

The converter rejects malformed triples, duplicate pairs, split overlap, and
non-contiguous global IDs rather than choosing a silent repair policy.
