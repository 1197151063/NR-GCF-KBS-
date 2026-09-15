# Efficiency Comparison

## Scope and measurement policy

This comparison uses the completed clean-data cases (`noise=0`, seed 2026) from the three formal 48-case experiments. LightGCN denotes the `lightgcn` arm, while SDR-GCF denotes the complete `full` arm with one-shot structure--dynamics filtering, synchronized graph/sampler reconstruction, and blended CrossNorm. GTN is an independent comparison method and has not produced logs yet.

- BPR: `/Users/chenyijun/Desktop/KBS2026/run (13).log` and `all_runs (10).json`
- SSM: `/Users/chenyijun/Desktop/KBS2026/run (14).log` and `all_runs (11).json`
- AU: `/Users/chenyijun/Desktop/KBS2026/run (15).log` and `all_runs (12).json`
- **Best epoch** is the epoch selected by Recall@20; its paired NDCG@20 is not used in this efficiency comparison.
- **Time to best** is the sum of the logged training time from epoch 1 through the best epoch.
- **Total training time** is the sum of all logged epoch training times until early stopping, including the 20-epoch patience period after the best epoch.
- **Time/epoch** is total training time divided by the number of executed epochs.
- The logged `time:` field covers the pre-evaluation training phase of each epoch. For SDR-GCF, the filtering-trigger epoch therefore also includes its one-shot reliability inference, graph reconstruction, and sampler reconstruction. It excludes evaluation, data preparation, process startup, and other end-to-end wall-clock costs.
- No inspected log records peak CPU RAM or peak GPU memory. Memory cells are therefore marked `--` rather than estimated. For LightGCN and SDR-GCF, `--` means **not measured**; for GTN, it additionally means **not run**.

## Complete clean-data comparison

All times are in seconds.

| Loss | Dataset | Method | Best epoch | Executed epochs | Time/epoch | Time to best | Total training time | Peak GPU memory (MiB) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BPR | LastFM | LightGCN | 417 | 437 | 0.168 | 70.29 | 73.54 | -- |
| BPR | LastFM | SDR-GCF | 92 | 112 | 0.373 | 35.36 | 41.75 | -- |
| BPR | LastFM | GTN | -- | -- | -- | -- | -- | -- |
| BPR | MovieLens-1M | LightGCN | 469 | 489 | 2.202 | 1032.96 | 1076.64 | -- |
| BPR | MovieLens-1M | SDR-GCF | 137 | 157 | 4.386 | 601.36 | 688.55 | -- |
| BPR | MovieLens-1M | GTN | -- | -- | -- | -- | -- | -- |
| BPR | Yelp2018 | LightGCN | 220 | 240 | 5.487 | 1210.31 | 1316.86 | -- |
| BPR | Yelp2018 | SDR-GCF | 36 | 56 | 7.324 | 273.98 | 410.14 | -- |
| BPR | Yelp2018 | GTN | -- | -- | -- | -- | -- | -- |
| BPR | Amazon-Book | LightGCN | 307 | 327 | 24.420 | 7506.85 | 7985.25 | -- |
| BPR | Amazon-Book | SDR-GCF | 74 | 94 | 28.250 | 2109.08 | 2655.51 | -- |
| BPR | Amazon-Book | GTN | -- | -- | -- | -- | -- | -- |
| SSM | LastFM | LightGCN | 64 | 84 | 0.156 | 9.84 | 13.13 | -- |
| SSM | LastFM | SDR-GCF | 65 | 85 | 0.382 | 25.82 | 32.49 | -- |
| SSM | LastFM | GTN | -- | -- | -- | -- | -- | -- |
| SSM | MovieLens-1M | LightGCN | 2 | 22 | 2.279 | 5.46 | 50.13 | -- |
| SSM | MovieLens-1M | SDR-GCF | 3 | 23 | 3.985 | 11.10 | 91.65 | -- |
| SSM | MovieLens-1M | GTN | -- | -- | -- | -- | -- | -- |
| SSM | Yelp2018 | LightGCN | 43 | 63 | 5.576 | 239.92 | 351.31 | -- |
| SSM | Yelp2018 | SDR-GCF | 2 | 22 | 8.386 | 20.07 | 184.50 | -- |
| SSM | Yelp2018 | GTN | -- | -- | -- | -- | -- | -- |
| SSM | Amazon-Book | LightGCN | 67 | 87 | 24.169 | 1619.35 | 2102.69 | -- |
| SSM | Amazon-Book | SDR-GCF | 1 | 21 | 29.664 | 33.91 | 622.95 | -- |
| SSM | Amazon-Book | GTN | -- | -- | -- | -- | -- | -- |
| AU | LastFM | LightGCN | 164 | 184 | 2.020 | 331.31 | 371.71 | -- |
| AU | LastFM | SDR-GCF | 17 | 37 | 2.287 | 40.81 | 84.61 | -- |
| AU | LastFM | GTN | -- | -- | -- | -- | -- | -- |
| AU | MovieLens-1M | LightGCN | 427 | 447 | 26.943 | 11504.41 | 12043.32 | -- |
| AU | MovieLens-1M | SDR-GCF | 72 | 92 | 27.446 | 1979.19 | 2525.03 | -- |
| AU | MovieLens-1M | GTN | -- | -- | -- | -- | -- | -- |
| AU | Yelp2018 | LightGCN | 106 | 126 | 44.555 | 4724.84 | 5613.94 | -- |
| AU | Yelp2018 | SDR-GCF | 20 | 40 | 43.615 | 886.86 | 1744.59 | -- |
| AU | Yelp2018 | GTN | -- | -- | -- | -- | -- | -- |
| AU | Amazon-Book | LightGCN | 87 | 107 | 99.085 | 8620.27 | 10602.09 | -- |
| AU | Amazon-Book | SDR-GCF | 116 | 136 | 97.693 | 11342.07 | 13286.25 | -- |
| AU | Amazon-Book | GTN | -- | -- | -- | -- | -- | -- |

## Main observations

1. **BPR:** SDR-GCF has a higher per-epoch cost on every dataset, but reaches its best epoch much earlier. Consequently, its total optimization time is lower on all four datasets: 43.2% lower on LastFM, 36.1% on MovieLens-1M, 68.9% on Yelp2018, and 66.7% on Amazon-Book.
2. **SSM:** SDR-GCF is faster overall on Yelp2018 (47.5% lower total training time) and Amazon-Book (70.4% lower), because the best performance occurs at epochs 2 and 1. It is slower on LastFM and MovieLens-1M, where the two methods execute nearly the same number of epochs while SDR-GCF has a larger per-epoch cost.
3. **AU:** Per-epoch costs are similar for the two methods on MovieLens-1M, Yelp2018, and Amazon-Book. SDR-GCF reduces total training time by 77.2% on LastFM, 79.0% on MovieLens-1M, and 68.9% on Yelp2018, but is 25.3% slower on Amazon-Book because its best epoch and stopping epoch are later.
4. **Objective cost matters:** AU is substantially more expensive per epoch than BPR/SSM, especially on MovieLens-1M, Yelp2018, and Amazon-Book. This is consistent with the batch-level pairwise-distance uniformity computation. Comparisons should therefore remain separated by objective rather than averaging BPR, SSM, and AU into one number.
5. **Memory remains unverified:** the current evidence supports timing and epoch comparisons only. A paper should not claim a numeric memory advantage until all three methods are rerun with the same peak-memory instrumentation and hardware.

## Paper-ready summary paragraph

We compare the training efficiency of LightGCN and SDR-GCF on the clean datasets under identical objective-specific protocols. The total training time is obtained by summing the pre-evaluation training-phase time of all epochs executed before early stopping; for SDR-GCF, this measurement includes the one-shot filtering and reconstruction work in the trigger epoch, while excluding evaluation and data preparation. Although SDR-GCF introduces additional computation for reliability estimation, graph reconstruction, and CrossNorm, its earlier convergence substantially reduces total training time in most settings. Under BPR, SDR-GCF lowers the total training time on all four datasets. Under AU, it is faster on LastFM, MovieLens-1M, and Yelp2018, but slower on Amazon-Book due to a later best epoch. Under SSM, the reduction is evident on Yelp2018 and Amazon-Book, whereas the additional per-epoch overhead dominates on LastFM and MovieLens-1M. Peak-memory usage was not recorded by the current runs and is therefore left unreported; GTN timing and memory entries will be filled after its experiments finish.

## Missing evidence to collect

For the final paper table, LightGCN, SDR-GCF, and GTN should be measured on the same GPU with the same software environment. Each run should record peak allocated GPU memory, peak reserved GPU memory, end-to-end wall-clock time, optimization-only epoch time, best epoch, and executed epochs. Peak-memory counters must be reset immediately before training and read after the final evaluation so that all methods use the same measurement boundary.
