import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go RecModel")
    
    parser.add_argument('--bpr_batch', type=int, default=1024,
                        help="the batch size for bpr loss training procedure")  # 512 1024 2048 4096

    parser.add_argument('--epochs', type=int, default=1000) 

    parser.add_argument('--testbatch', type=int, default=512,
                        help="the batch size of users for testing")

    parser.add_argument('--seed', type=int, default=0 ,help='random seed')

    parser.add_argument('--K', type=int, default=3)

    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate:0.001")  # 0.001
    
    parser.add_argument('--dataset', type=str, default='yelp2018')

    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help=(
            'global early-stopping patience monitored on Recall@20; '
            'training stops after this many consecutive epochs without '
            'a strict Recall@20 improvement (default: 20)'
        ),
    )

    parser.add_argument('--dropout', type=float, default=1e-1)

    parser.add_argument('--num_neg', type=int, default=64)

    parser.add_argument('--init_weight', type=float, default=1.0)

    parser.add_argument('--decay', type=float, default=1e-4)

    parser.add_argument('--tau', type=float, default=0.1,
                        help="the temperature for softmax in loss function")  # 0.1
    parser.add_argument('--lambda_', type=float, default=1,
                        help="legacy compatibility parameter; NRGCF stage-two direct cross_norm does not use lambda")
    parser.add_argument(
        '--representation-modulation-mode',
        type=str,
        default='original_stage_two',
        choices=[
            'none',
            'legacy_always',
            'original_always',
            'original_stage_two',
            'paper_stage_two',
            'reliability_weighted_always',
            'reliability_weighted_stage_two',
        ],
        help=(
            'none disables modulation; legacy_always/original_always apply '
            'the supplied direct cross_norm from epoch one; original_stage_two activates the '
            'released direct cross_norm operation only after filtering; '
            'paper_stage_two is a backward-compatible alias; '
            'reliability_weighted_always keeps cross_norm active throughout '
            'and changes only its RMS estimator after filtering; '
            'reliability_weighted_stage_two additionally estimates the '
            'stage-two scales from frozen retained-edge reliability'
        ),
    )
    parser.add_argument(
        '--representation-modulation-ramp-epochs',
        type=int,
        default=0,
        help=(
            'number of post-filter epochs used to linearly introduce '
            'stage-two modulation; 0 is the exact hard stage transition'
        ),
    )

    parser.add_argument('--export-edge-diagnostics', action='store_true',
                        help='export side-channel per-edge diagnostics at the current filtering point')
    parser.add_argument('--edge-diagnostics-dir', type=str, default='edge_diagnostics',
                        help='diagnostics output directory (default: ./edge_diagnostics)')
    parser.add_argument('--edge-diagnostics-format', type=str, default='parquet',
                        choices=['parquet', 'csv', 'csv_gzip'],
                        help='streaming diagnostics format; parquet falls back to one gzip CSV without pyarrow')
    parser.add_argument('--edge-diagnostics-structural-mode', type=str,
                        default='two_hop_minhash',
                        choices=['two_hop_minhash', 'none'],
                        help='training-graph-only structural feature mode')
    parser.add_argument('--edge-diagnostics-topk', type=int, default=10,
                        help='top-k for bounded structural neighbor summaries')
    parser.add_argument('--edge-diagnostics-chunk-size', type=int, default=8192,
                        help='number of edge rows computed per streaming write')
    parser.add_argument('--edge-diagnostics-labels-file', type=str, default=None,
                        help='optional ordered synthetic-label CSV used only during diagnostics export')
    parser.add_argument('--edge-diagnostics-noise-validation-file', type=str, default=None,
                        help='optional validated noise metadata JSON copied into diagnostics provenance')
    parser.add_argument('--edge-diagnostics-verify-invariance', action='store_true',
                        help='verify exporter leaves tracked tensors, parameters, and RNG state unchanged')
    parser.add_argument('--edge-diagnostics-min-degree', type=int, default=2,
                        help='transparent post-removal minimum-degree risk threshold')
    parser.add_argument('--edge-diagnostics-stop-after-filter', action='store_true',
                        help='explicit smoke-test option: stop after the epoch-15 filtering/export point')
    parser.add_argument('--requested-noise-ratio', type=float, default=None,
                        help='metadata only; does not inject noise or assert an actual noise ratio')

    parser.add_argument('--edge-filter-mode', type=str, default='current',
                        choices=['current', 'none', 'hard_consensus',
                                 'hard_structure_only', 'soft_reliability',
                                 'gated_soft_reliability',
                                 'hard_structure_momentum'],
                        help='graph filtering policy; current preserves the original epoch-15 NR-GCF implementation')
    parser.add_argument('--export-edge-reliability-summary', action='store_true',
                        help='write compact JSON policy statistics without a per-edge table')
    parser.add_argument('--edge-reliability-dir', type=str, default='edge_reliability',
                        help='compact JSON output directory for none/hard/soft reliability comparisons')
    parser.add_argument('--edge-reliability-labels-file', type=str, default=None,
                        help='optional ordered synthetic-label CSV used only for compact post-decision statistics')
    parser.add_argument('--edge-reliability-noise-validation-file', type=str, default=None,
                        help='optional validated noise metadata JSON copied into compact reliability provenance')
    parser.add_argument('--edge-reliability-momentum-quantile', type=float, default=0.80,
                        help='hard consensus high-momentum percentile threshold')
    parser.add_argument('--edge-reliability-structure-quantile', type=float, default=0.20,
                        help='hard consensus low-structure percentile threshold')
    parser.add_argument('--edge-reliability-structure-weight', type=float, default=0.95,
                        help='diagnostic soft reliability weight assigned to structural percentile rank')
    parser.add_argument('--edge-reliability-min-weight', type=float, default=0.10,
                        help='minimum propagation weight in soft reliability mode')
    parser.add_argument('--edge-reliability-filtering-epoch', type=int, default=15,
                        help='warm-up/filter epoch for hard_structure_momentum')
    parser.add_argument('--edge-reliability-momentum-decay', type=float, default=0.90,
                        help='stable per-edge EMA decay for hard_structure_momentum')

    return parser.parse_args()
