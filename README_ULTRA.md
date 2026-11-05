# ULTRA: Inductive Link Prediction on Knowledge Graphs

## 1. Background and Model Overview

Knowledge graph (KG) link prediction aims to infer missing facts represented as triples $(h,r,t)$, where $h$ is the head entity, $r$ is the relation type, and $t$ is the tail entity. Given a query $(h,r,?)$, ULTRA ranks candidate tail entities to identify plausible missing facts. It also supports head prediction $(?,r,t)$ by converting it into an equivalent tail-prediction query $(t,r^{-1},?)$ using the inverse relation $r^{-1}$. Both tasks are therefore handled through tail-entity scoring.

ULTRA is designed for inductive knowledge graph reasoning: a pretrained model can be applied to new knowledge graphs with previously unseen entities and potentially different relation vocabularies. Rather than relying on entity- or relation-specific embeddings tied to a particular graph, it learns transferable structural patterns based on how relations interact.

The model uses two message-passing stages. A **relation encoder** computes query-conditioned representations of relation types on a relation graph. An **entity encoder** then uses these representations to propagate information over the entity graph and assign scores to candidate entities. This project loads a pretrained ULTRA checkpoint and evaluates its zero-shot link-prediction performance on inductive benchmarks; it does not retrain the model on the evaluation datasets.

## 2. Relation Graph Construction

A knowledge graph is represented as $G=(V,E,R)$, where $V$ is the entity set, $R$ is the set of relation types, and $E\subseteq V\times R\times V$ is the set of directed, relation-typed triples. A triple $(u,r,v)$ denotes a directed edge from head entity $u$ to tail entity $v$ with relation type $r$.

Before relation graph construction, each original triple $(u,r,v)$ is supplemented with an inverse triple $(v,r^{-1},u)$. If the original graph has $R_0$ relation types and each is assigned a distinct inverse relation, the augmented relation set is

$$
R_{\mathrm{aug}}=R_{\mathrm{direct}}\cup R_{\mathrm{inverse}},
\qquad |R_{\mathrm{aug}}|=2R_0.
$$

In the following, $E$ and $R$ refer to the **augmented** triple set and relation set, respectively. The same augmented relation graph supports both head and tail prediction; separate relation graphs are not constructed for the two prediction directions.

ULTRA constructs a relation graph $L(G)=(V_L,E_L,R_L)$ in which each relation type (including each inverse relation) becomes a node:

$$
V_L=R,\qquad
R_L=\{\mathrm{t2h},\mathrm{h2h},\mathrm{h2t},\mathrm{t2t}\},
\qquad E_L\subseteq R\times R_L\times R.
$$

Thus, $|V_L|=2R_0$, while $|R_L|=4$. The four types in $R_L$ describe **structural connections between relation nodes**, not the original KG's relation labels. A relation-graph edge $(r_1,\tau,r_2)$ exists when triples involving $r_1$ and $r_2$ share an entity in the endpoint configuration specified by $\tau$:

$$
\begin{aligned}
\mathrm{t2h}:&\quad (u,r_1,v)\in E,\quad (v,r_2,w)\in E,\\
\mathrm{h2h}:&\quad (v,r_1,u)\in E,\quad (v,r_2,w)\in E,\\
\mathrm{h2t}:&\quad (v,r_1,u)\in E,\quad (w,r_2,v)\in E,\\
\mathrm{t2t}:&\quad (u,r_1,v)\in E,\quad (w,r_2,v)\in E.
\end{aligned}
$$

The four patterns can be visualized as follows (arrows show the directions of the **original KG triples**):

```text
t2h:  (u) --r1--> (v) --r2--> (w)

h2h:  (u) <--r1-- (v) --r2--> (w)

h2t:  (u) <--r1-- (v) <--r2-- (w)

t2t:  (u) --r1--> (v) <--r2-- (w)
```

These structural connections allow ULTRA to represent relation types through their interactions, even when different knowledge graphs use unrelated relation vocabularies.

## 3. ULTRA Model

ULTRA performs query-conditioned message passing at two levels: first over the relation graph, then over the entity graph. The following equations describe the sum-aggregation, element-wise multiplicative message passing, LayerNorm, ReLU, and residual connections used in the implementation.

### 3.1. Query-Conditioned Relation Representation Learning

For a query relation $q\in R$, the relation encoder initializes relation node $r$ with

$$
\mathbf h_{r\mid q}^{(0)}=
\begin{cases}
\mathbf 1_d, & r=q,\\
\mathbf 0_d, & r\ne q.
\end{cases}
$$

Each structural edge type $\tau\in R_L$ has a learnable embedding $\mathbf z_{\tau}^{(\ell)}\in\mathbb R^d$ at message-passing layer $\ell$. An edge $(r',\tau,r)$ sends the message

$$
\mathbf m_{r'\to r}^{(\ell)}
=\mathbf h_{r'\mid q}^{(\ell)}\odot\mathbf z_{\tau}^{(\ell)},
$$

where $\odot$ denotes element-wise multiplication. Incoming messages are summed, and the initial query boundary feature is injected into the aggregation:

$$
\mathbf a_r^{(\ell)}
=\mathbf h_{r\mid q}^{(0)}+
\sum_{(r',\tau,r)\in E_L}
\mathbf h_{r'\mid q}^{(\ell)}\odot\mathbf z_{\tau}^{(\ell)}.
$$

The layer then combines the **current** relation-node representation with the aggregated features:

$$
\widetilde{\mathbf h}_{r\mid q}^{(\ell+1)}
=\operatorname{ReLU}\!\left(
\operatorname{LN}\!\left(
W_R^{(\ell)}
\left[\mathbf h_{r\mid q}^{(\ell)}\parallel\mathbf a_r^{(\ell)}\right]
+\mathbf b_R^{(\ell)}
\right)\right),
$$

$$
\mathbf h_{r\mid q}^{(\ell+1)}
=\mathbf h_{r\mid q}^{(\ell)}+
\widetilde{\mathbf h}_{r\mid q}^{(\ell+1)}.
$$

Here, $\parallel$ denotes concatenation and $\operatorname{LN}$ denotes LayerNorm. The residual addition is applied when the input and output hidden-state shapes match. In particular, the concatenated node representation is $\mathbf h_{r\mid q}^{(\ell)}$, **not** the initial state $\mathbf h_{r\mid q}^{(0)}$; the initial state instead appears in the aggregated term as a boundary feature.

After $T$ relation message-passing layers, the encoder produces query-conditioned representations $\{\mathbf h_{r\mid q}^{(T)}:r\in R\}$ for all relation types. These representations are computed for the current query relation rather than retrieved from a fixed, graph-specific relation embedding table.

### 3.2. Query-Conditioned Entity Representation Learning

For a query $(h,q,?)$, the entity encoder uses the relation encoder's final representation of $q$ to initialize the source entity. Other entities have zero initial boundary features:

$$
\mathbf h_{v\mid h,q}^{(0)}=
\begin{cases}
\mathbf h_{q\mid q}^{(T)}, & v=h,\\
\mathbf 0_d, & v\ne h.
\end{cases}
$$

For an entity-graph edge $(w,r,v)\in E$, the message from entity $w$ to entity $v$ at layer $\ell$ is

$$
\mathbf m_{w\to v}^{(\ell)}
=\mathbf h_{w\mid h,q}^{(\ell)}
\odot\operatorname{MLP}^{(\ell)}\!\left(
\mathbf h_{r\mid q}^{(T)}\right).
$$

The message combines the sender's current representation with a learned projection of the **query-conditioned** representation of the edge's relation type. Aggregation runs over all incoming relation-typed edges and incorporates the initial boundary feature:

$$
\mathbf a_v^{(\ell)}
=\mathbf h_{v\mid h,q}^{(0)}+
\sum_{(w,r,v)\in E}
\mathbf h_{w\mid h,q}^{(\ell)}
\odot\operatorname{MLP}^{(\ell)}\!\left(
\mathbf h_{r\mid q}^{(T)}\right).
$$

The entity-node update is

$$
\widetilde{\mathbf h}_{v\mid h,q}^{(\ell+1)}
=\operatorname{ReLU}\!\left(
\operatorname{LN}\!\left(
W_E^{(\ell)}
\left[\mathbf h_{v\mid h,q}^{(\ell)}\parallel\mathbf a_v^{(\ell)}\right]
+\mathbf b_E^{(\ell)}
\right)\right),
$$

$$
\mathbf h_{v\mid h,q}^{(\ell+1)}
=\mathbf h_{v\mid h,q}^{(\ell)}+
\widetilde{\mathbf h}_{v\mid h,q}^{(\ell+1)}.
$$

As in the relation encoder, the residual addition applies when the hidden-state shapes match. The concatenation uses the current layer's entity representation, while the initial boundary feature enters through aggregation.

After $L$ entity message-passing layers, a candidate tail $t$ is scored using its final entity representation and the **final**, query-conditioned representation of relation $q$:

$$
s(h,q,t)
=\operatorname{MLP}_{\mathrm{score}}\!\left(
\mathbf h_{t\mid h,q}^{(L)}\parallel
\mathbf h_{q\mid q}^{(T)}
\right).
$$

The scoring MLP outputs a scalar score (logit). Larger scores indicate more plausible candidate triples; they are not necessarily calibrated probabilities.

## 4. Evaluation Datasets

The notebook evaluates ULTRA on eight inductive KG link-prediction benchmarks covering different generalization settings:

- **NELLInductive:** `v1`, `v2`, `v3`, and `v4`, derived from NELL. These benchmarks emphasize generalization to previously unseen entities.
- **WKIngram:** `25`, `50`, `75`, and `100`, derived from Wikidata and provided in the InGram benchmark format. These benchmarks test generalization to new entities and relation vocabularies.

The notebook loads the pretrained checkpoint `3g_reproduced.pth`, pretrained on **FB15k-237**, **WN18RR**, and **CoDEx-Medium**, and evaluates it without training or fine-tuning on the eight target benchmark datasets. The results therefore measure **zero-shot inductive link-prediction performance**.

## 5. Filtered Ranking Evaluation

For each test triple $(h,r,t)$, the evaluator computes a filtered rank for both head and tail prediction.

A query may have more than one correct answer. Under the **filtered** evaluation protocol, other known true answers are excluded from the negative candidate set, so the model is not penalized for assigning them high scores. The current target is scored separately and is **not** included among the negative candidates.

Let $V$ be the candidate entity set and let $E_{\mathrm{filter}}$ contain the known true triples used for filtering. For a tail-prediction query $(h,r,?)$, define the set of known correct tails as

$$
T^+(h,r)=\{x\in V:(h,r,x)\in E_{\mathrm{filter}}\},
$$

and the filtered negative candidate set as

$$
T^-(h,r)=V\setminus T^+(h,r).
$$

For the correct target $t$, the filtered rank is

$$
\boxed{
\operatorname{rank}(t)
=1+\sum_{x\in T^-(h,r)}
\mathbb I\!\left[s(h,r,x)\ge s(h,r,t)\right]
}
$$

where $\mathbb I[\cdot]$ is an indicator function. The use of $\ge$ implements **pessimistic tie-breaking**: a remaining negative candidate with the same score as the true target is counted as ranking at least as highly as the target. Known true alternatives that have been filtered out do not affect the rank.

The filtering graph is used to construct evaluation masks and is distinct from the graph used for neural message passing. In particular, including validation or test target triples in the **filtering graph** does not mean that those triples are provided to the model as message-passing edges. The specific combination of inference, validation, and test triples used for filtering follows the dataset-specific evaluation setup in the notebook.

## 6. Metrics and Results

Each test triple contributes a head-prediction rank and a tail-prediction rank. If there are $M$ evaluated test triples and both directions are included, the evaluator obtains $N=2M$ ranks. The reported metrics are calculated over these filtered ranks:

$$
\mathrm{MR}=\frac{1}{N}\sum_{i=1}^{N}\operatorname{rank}_i,
\qquad
\mathrm{MRR}=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{\operatorname{rank}_i},
$$

$$
\mathrm{Hits@}k
=\frac{1}{N}\sum_{i=1}^{N}
\mathbb I\!\left[\operatorname{rank}_i\le k\right],
\qquad k\in\{1,3,10\}.
$$

**Lower** Mean Rank (MR) is better; **higher** Mean Reciprocal Rank (MRR) and Hits@k are better. All results below use **filtered ranking**.

| Dataset | MR ↓ | MRR ↑ | Hits@1 ↑ | Hits@3 ↑ | Hits@10 ↑ |
|---|---:|---:|---:|---:|---:|
| NELLInductive (v1) | 3.7189 | 0.7071 | 0.6517 | 0.6791 | 0.9279 |
| NELLInductive (v2) | 62.5481 | 0.5086 | 0.3920 | 0.5813 | 0.7150 |
| NELLInductive (v3) | 62.6997 | 0.4963 | 0.3951 | 0.5414 | 0.6954 |
| NELLInductive (v4) | 82.6389 | 0.4530 | 0.3331 | 0.5214 | 0.6835 |
| WKIngram (25) | 163.8585 | 0.2635 | 0.1569 | 0.3156 | 0.4730 |
| WKIngram (50) | 617.6747 | 0.1494 | 0.0814 | 0.1550 | 0.2930 |
| WKIngram (75) | 139.1691 | 0.3446 | 0.2535 | 0.4012 | 0.5022 |
| WKIngram (100) | 637.7028 | 0.1613 | 0.1032 | 0.1777 | 0.2748 |

## 7. Reproducibility Notes

- **Evaluation mode:** Zero-shot inference using the pretrained `3g_reproduced.pth` checkpoint; no downstream training or fine-tuning.
- **Random seed:** The evaluation code sets the PyTorch random seed to `1024`.
- **Model configuration:** Both relation and entity encoders use `hidden_dims = [64] * 7` (seven 64-dimensional message-passing layers).
- **Evaluation batch size:** 16 test triples per batch.
- **Device:** CUDA when available, otherwise CPU.
- **Data and checkpoint access:** Dataset files and the pretrained checkpoint are loaded from the paths specified in the notebook.

## References

- Galkin, M., Yuan, X., Mostafa, H., Tang, J., and Zhu, Z. **Towards Foundation Models for Knowledge Graph Reasoning.** ICLR 2024. [Paper](https://arxiv.org/abs/2310.04562).
- [Official ULTRA implementation (PyTorch Geometric)](https://github.com/DeepGraphLearning/ULTRA).
