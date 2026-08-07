# Official Implementation of NR-GCF

NR-GCF is a noise resistant training diagram  for graph collaborative filtering. This repo provides the official open-source implementation of our paper:

+ Yijun Chen, Bohan Li, Yicong Li, et al. NR-GCF: Graph Collaborative Filtering with Improved Noise Resistance. In CIKM 2025.

## Environments

torch_geometric==2.5.3

numpy==1.24.3

scipy==1.10.1

torch-sparse==0.6.17+pt20cu118

torch==2.0.1

## Main Result Under Conventional Public Benchmark datasets

#### Optimize with BPR loss

**Results on Yelp2018**

| BackBone Model | Recall@20  | NDCG@20    |
| -------------- | ---------- | ---------- |
| LightGCN       | 0.0639     | 0.0525     |
| GTN(d=64)      | 0.0637     | 0.0520     |
| LightGCN+NRGCF | **0.0679** | **0.0561** |

**Results on Amazon-book**

| BackBone Model | Recall@20  | NDCG@20    |
| -------------- | ---------- | ---------- |
| LightGCN       | 0.0411     | 0.0318     |
| GTN(d=64)      | 0.0404     | 0.0312     |
| LightGCN+NRGCF | **0.0467** | **0.0366** |



#### Optimize with InfoNCE loss

**Results on Yelp2018**

| BackBone Model | Recall@20  | NDCG@20    |
| -------------- | ---------- | ---------- |
| SGL            | 0.0671     | 0.0549     |
| RGCF           | 0.0665     | 0.0547     |
| SGL + NRGCF    | **0.0694** | **0.0568** |

**Results on Amazon-book**

| BackBone Model | Recall@20  | NDCG@20    |
| -------------- | ---------- | ---------- |
| SGL            | 0.0472     | 0.0374     |
| RGCF           | 0.0461     | 0.0367     |
| SGL+NRGCF      | **0.0494** | **0.0387** |



#### Optimize with A&U loss

**Results on Yelp2018**

| BackBone Model | Recall@20  | NDCG@20    |
| -------------- | ---------- | ---------- |
| DirectAU       | 0.0701     | 0.0595     |
| BOD            | 0.0700     | 0.0588     |
| DirectAU+NRGCF | **0.0725** | **0.0601** |

**Results on Amazon-book**

| BackBone Model | Recall@20  | NDCG@20    |
| -------------- | ---------- | ---------- |
| DirectAU       | 0.0435     | 0.0351     |
| BOD            | 0.0424     | 0.0339     |
| DirectAU+NRGCF | **0.0457** | **0.0362** |


## How to run 

Either modify the command in 

> run_bash.sh 

or copy it out and execute it manually:

> python NR-GCF.py  --dataset yelp2018  --lr 5e-4  --init_weight 0.01 --patience 20 --lambda_ 0.6

## Converted NT-SSM datasets

The repository also contains `lastfm` and `ml-1m` in NR-GCF's grouped format.
For both datasets, the released NT-SSM train and validation splits are merged
for training while every released test interaction is kept in test.  Conversion
counts, hashes, ID checks, and the retained test cold-start items are documented
in [`data/NTSSM_CONVERSION.md`](data/NTSSM_CONVERSION.md) and each dataset's
`conversion_metadata.json`.

The one-seed clean/20%-replacement transfer pilot can be launched with:

```bash
bash code/run_ntssm_dataset_generalization_v2_3.sh
```
