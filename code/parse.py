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

    parser.add_argument('--patience', type=int, default=50)

    parser.add_argument('--dropout', type=float, default=1e-1)

    parser.add_argument('--num_neg', type=int, default=64)

    parser.add_argument('--init_weight', type=float, default=1.0)

    parser.add_argument('--decay', type=float, default=1e-4)

    parser.add_argument('--tau', type=float, default=0.1,
                        help="the temperature for softmax in loss function")  # 0.1
    parser.add_argument('--lambda_', type=float, default=1,
                        help="the lambda for cross norm in loss function")  # 0.5

    parser.add_argument('--export-edge-diagnostics', action='store_true',
                        help='export side-channel per-edge diagnostics at the current filtering point')
    parser.add_argument('--edge-diagnostics-dir', type=str, default='edge_diagnostics',
                        help='diagnostics output directory (default: ./edge_diagnostics)')
    parser.add_argument('--edge-diagnostics-format', type=str, default='parquet',
                        choices=['parquet', 'csv'],
                        help='preferred diagnostics part format; parquet falls back to CSV without pyarrow')
    parser.add_argument('--edge-diagnostics-structural-mode', type=str,
                        default='two_hop_countsketch',
                        choices=['two_hop_countsketch', 'none'],
                        help='training-graph-only structural feature mode')
    parser.add_argument('--edge-diagnostics-topk', type=int, default=10,
                        help='top-k for bounded structural neighbor summaries')
    parser.add_argument('--edge-diagnostics-chunk-size', type=int, default=65536,
                        help='number of edge rows computed and written per diagnostics part')
    parser.add_argument('--edge-diagnostics-verify-invariance', action='store_true',
                        help='verify exporter leaves tracked tensors, parameters, and RNG state unchanged')
    parser.add_argument('--edge-diagnostics-min-degree', type=int, default=2,
                        help='transparent post-removal minimum-degree risk threshold')
    parser.add_argument('--edge-diagnostics-stop-after-filter', action='store_true',
                        help='explicit smoke-test option: stop after the epoch-15 filtering/export point')
    parser.add_argument('--requested-noise-ratio', type=float, default=None,
                        help='metadata only; does not inject noise or assert an actual noise ratio')

    return parser.parse_args()
