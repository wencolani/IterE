# IterE

## INTRODUCTION

Iteratively Learning Embeddings and Rules for Knowledge Graph Reasoning. (WWW'19)

## RUN

python3 main.py --device /gpu --datadir ./datasets/FB15k-237-sparse --batch_size 2048 --dim 200 --test_batch_size 50 --max_epoch 100 --test_per_iter 100 --num_test 3000 --axiom_weight 0.1 --optimize Adam --lr 0.001  --neg_samples 2 --regularizer_weight 0.00001 --save_dir ./save/0825AE1  --update_axiom_per 1  --axiom_probability 0.95 --triple_generator 3 

## DATASET

There are four sparse datasets used in this paper, WN18-sparse, FB15k-sparse WN18-sparse and FB15k-237-sparse, which are in folder ./data. We provide the axiom pool version for each dataset together with their entailments.

We also provide the code for generating axiom pools, please refer to axiomPools.py if you are interested in axiom pool genration or apply them to other datasets.

## CITE

If the codes help you or the paper inspire your, please cite following paper:

Wen Zhang, Bibek Paudel, Liang Wang, Jiaoyan Chen, Hai Zhu, Wei Zhang,  Abraham Bernstein and Huajun Chen. Iteratively Learning Embeddings and Rules for Knowledge Graph Reasoning. In proceedings of the 2019 World Wide Web Conference (WWW'19).

