
<p align="center">
    <a href="https://github.com/zjunlp/openue"> <img src="https://raw.githubusercontent.com/zjunlp/openue/master/docs/images/logo_zju_klab.png" width="400"/></a>
</p>

<p align="center">
  	<strong>IterE: a knowledge graph reasoning method iteratively learning rules and embeddings.</strong>
    </font>
</p>


This repository is the official introduction of **[Iteratively Learning Embeddings and Rules for Knowledge Graph Reasoning](https://dl.acm.org/doi/10.1145/3308558.3313612)** . This paper has been accepted by **WWW 2019** main conference. 


# Brief Introduction


### Abstract

Reasoning is essential for the development of large knowledge graphs, especially for completion, which aims to infer new triples based on existing ones. Both rules and embeddings can be used for knowledge graph reasoning and they have their own advantages and difficulties. Rule-based reasoning is accurate and explainable but rule learning with searching over the graph always suffers from efficiency due to huge search space. Embedding-based reasoning is more scalable and efficient as the reasoning is conducted via computation between embeddings, but it has difficulty learning good representations for sparse entities because a good embedding relies heavily on data richness. Based on this observation, in this paper we explore how embedding and rule learning can be combined together and complement each other's difficulties with their advantages. We propose a novel framework IterE iteratively learning embeddings and rules, in which rules are learned from embeddings with proper pruning strategy and embeddings are learned from existing triples and new triples inferred by rules. Evaluations on embedding qualities of IterE show that rules help improve the quality of sparse entity embeddings and their link prediction results. We also evaluate the efficiency of rule learning and quality of rules from IterE compared with AMIE+, showing that IterE is capable of generating high quality rules more efficiently. Experiments show that iteratively learning embeddings and rules benefit each other during learning and prediction.


### Model

Our research question in this paper is whether it is possible to learn embeddings and rules at the same time and make their advantages complement to each other's difficulties. 

In this paper, we propose a novel framework IterE that iteratively learns embeddings and rules, which can combine many embedding methods and different kinds of rules. Especially, we consider linear map assumption for embedding learning because it is inherently friendly for rule learning as there are special rule conclusions for relation embeddings in rules.  We also consider a particular collection of object property axioms defined in OWL2 for rule learning considering that semantics included in web ontology language are important for the development of knowledge graph.

<img src="figures/axioms.png" alt="axioms.png" style="zoom:100%;" />

Rule conclusion in above table is the main supports for IterE. IterE mainly includes three parts: (1) embedding learning, (2) axiom induction, and (3) axiom injection as follow  

<img src="./figures/IterE.jpg" alt="IterE.jpg" style="zoom:30%;" />

* **Embedding learning** learns entity embeddings E and relation embeddings R with a loss function Lembedd in д to be minimized, calculated with input triples (s,r, o), each with a label related with its truth value. The inputs are of two types: triples existing in K and triples that do not exist in K but are inferred by axioms.
* **Axiom Induction** inducts a set of axioms A based on relation embeddings R from the embedding learning step, and assigns each axiom with a score s_{axiom}.
* **Axiom Injection** injects new triples about sparse entities in K to help improve their poor embeddings caused by insufficient training. The new triples are inferred from groundings of quality axioms with high scores in A from axiom induction. After axiom injection, with K updated, the process goes back to embedding learning again. 

### Experiments

We experimented on 4 datasets and proved that:

* Through link prediction experiments, we proved that : (1) by injecting new triples for sparse entities, axioms help improve the quality of sparse entity embeddings and are more helpful in sparse KGs. (2) Combining axioms and embeddings together to predict missing links works better than using embeddings only. Both the deductive capability of axioms and the inductive capability of embeddings contribute to prediction and complement each other.  

  <img src="figures/experiment1.jpg" alt="experiment1.jpg" style="zoom:50%;" />

* Through rule learning experiments, we can proved that (1) embeddings together with axiom pool generation help rule learning overcome large search space problem and improve rule learning efficiency, and (2) they also improve rule learning qualities and rules’ reliable scores generated based on calculation with embeddings.

  <img src="figures/experiment2.jpg" alt="experiment2.jpg" style="zoom:50%;" />

* Through dive into details of iteratively learning, we proved that: (1) Iterative learning benefits embedding learning as the quality of embeddings gets better during training. (2) Iteratively learning benefits axiom learning because more axioms are learned and more triples are injected during training. (3) Axioms and embeddings influence and constrain each other during training. 

<img src="/Users/wen/Documents/1事项/202108-github开源readme准备/IterE README/figures/experiment3.jpg" alt="experiment3.jpg" style="zoom:50%;" />

# Use the code

### Requirements

This is implemented with Tensorflow 1.X 

### Training and Evaluation

To train and evaluate the model(s) in the paper, run this command:

```python
python3 main.py --device /gpu --datadir ./datasets/FB15k-237-sparse --batch_size 2048 --dim 200 --test_batch_size 50 --max_epoch 100 --test_per_iter 100 --num_test 3000 --axiom_weight 0.1 --optimize Adam --lr 0.001 --neg_samples 2 --regularizer_weight 0.00001 --save_dir ./save/0825AE1 --update_axiom_per 1 --axiom_probability 0.95 --triple_generator 3
```

# How to Cite

If you use or extend our work, please cite the following paper:

```
@inproceedings{IterE,
  author    = {Wen Zhang and
               Bibek Paudel and
               Liang Wang and
               Jiaoyan Chen and
               Hai Zhu and
               Wei Zhang and
               Abraham Bernstein and
               Huajun Chen},
  title     = {Iteratively Learning Embeddings and Rules for Knowledge Graph Reasoning},
  booktitle = {{WWW}},
  pages     = {2366--2377},
  publisher = {{ACM}},
  year      = {2019}
}
```
