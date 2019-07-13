# Deep Graph Convolutional Networks

## GCN
### Node Classification
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (GCN)
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (GraphSAGE)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (GAT)
- [Representation Learning on Graphs with Jumping Knowledge Networks](https://arxiv.org/abs/1806.03536)
- [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153) (SGC)
	- [Code (PyTorch)](https://github.com/Tiiiger/SGC)
- [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://openreview.net/forum?id=H1gL-2A9Ym) (APPNP)
- [Deep Graph Infomax](https://arxiv.org/pdf/1809.10341.pdf)
- [MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing](https://arxiv.org/abs/1905.00067)
- [Position-aware Graph Neural Networks](http://proceedings.mlr.press/v97/you19b.html)
- [Disentangled Graph Convolutional Networks](http://proceedings.mlr.press/v97/ma19a/ma19a.pdf)
- [A Representation Learning Framework for Property Graphs](https://yaobaiwei.github.io/papers/PGE_KDD19.pdf)
- [Exploiting Edge Features for Graph Neural Networks](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gong_Exploiting_Edge_Features_for_Graph_Neural_Networks_CVPR_2019_paper.pdf)
- [Power up! Robust Graph Convolutional Network against Evasion Attacks based on Graph Powering](https://arxiv.org/pdf/1905.10029.pdf)
	- Read carefully. Re-examine Laplacian operator - find some basic flaws in the spatial and spectral domains. Propose a new operator.

### Graph Classification
- [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) (GIN)
- [An End-to-End Deep Learning Architecture for Graph Classiﬁcation](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf) (DGCNN)
	- [Code (Torch)](https://github.com/muhanzhang/DGCNN), [Code (PyTorch)](https://github.com/muhanzhang/pytorch_DGCNN)
- [Graph Capsule Convolutional Neural Networks](https://arxiv.org/abs/1805.08090)
- [Capsule Graph Neural Network](https://openreview.net/forum?id=Byl8BnRcYm)
	- [Code (PyTorch)](https://github.com/benedekrozemberczki/CapsGNN), [Code (TF)](https://github.com/XinyiZ001/CapsGNN)
- [Discriminative structural graph classification](https://arxiv.org/pdf/1905.13422.pdf)
	- Discrimination capacity of aggregation functions
- [Relational Pooling for Graph Representations](https://arxiv.org/abs/1903.02541)
	- Read carefully. Use node and edge features.
- [Optimal Transport for structured data with application on graphs](http://proceedings.mlr.press/v97/titouan19a/titouan19a.pdf)
	- See [Wasserstein Weisfeiler-Lehman Graph Kernels](https://arxiv.org/pdf/1906.01277.pdf)

### Graph Learning
- [Learning Discrete Structures for Graph Neural Networks](https://arxiv.org/abs/1903.11960) 
	- Sample graphs, learn distribution of each graph, and node classification. Transductive.
- [Exploring Graph Learning for Semi-Supervised Classification Beyond Euclidean Data](https://arxiv.org/abs/1904.10146) 
	- Learn adj matrix.
- [Graph Learning Networks](https://graphreason.github.io/papers/7.pdf) (New problem)
	- Build an explicit graph and learn the graph.
- [Graph Matching Networks for Learning the Similarity of Graph Structured Objects](https://arxiv.org/abs/1904.12787)
	- Graph similarity.
- [Learning to Route in Similarity Graphs](http://proceedings.mlr.press/v97/baranchuk19a/baranchuk19a.pdf)
	- Routing problem in graphs. Neareast neighbor search.
- [Large Scale Graph Learning From Smooth Signals](https://openreview.net/forum?id=ryGkSo0qYm)
	- Graph construction by approximate neareat neighbor techniques. Make it efficient.
- [Spectral Inference Networks: Unifying Deep and Spectral Learning](https://arxiv.org/abs/1806.02215)

### Unsupervised Learning
- [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/pdf/1811.12359.pdf)
- [Pre-training Graph Neural Networks](https://arxiv.org/pdf/1905.12265.pdf)
- [Disentangled Graph Convolutional Networks](http://proceedings.mlr.press/v97/ma19a/ma19a.pdf)
- [Deep Graph Infomax](https://arxiv.org/pdf/1809.10341.pdf)
- [UNSUPERVISED PRE-TRAINING OF GRAPH CONVOLUTIONAL NETWORKS](https://acbull.github.io/pdf/iclr19-pretrain.pdf)
- [Graphite: Iterative Generative Modeling of Graphs](https://arxiv.org/abs/1803.10459)
	- An algorithmic framework for unsupervised learning of representations over nodes in large graphs using deep latent variable generative models.

### Subgraph Embeddings

### Self-attention CNN
- [Stand-Alone Self-Attention in Vision Models](https://arxiv.org/pdf/1906.05909.pdf)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (GAT)

### Position-aware GCN
- [Position-aware Graph Neural Networks](http://proceedings.mlr.press/v97/you19b.html)

### Adversarial Attacks
- [Attacking Graph Convolutional Networks via Rewiring](https://arxiv.org/pdf/1906.03750.pdf) (New problem)
	- Rewiring attack. Perturbation
- [Robust Graph Convolutional Networks Against Adversarial Attacks](http://pengcui.thumedialab.com/papers/RGCN.pdf)
- [Stability Properties of Graph Neural Networks](https://arxiv.org/pdf/1905.04497.pdf)
	- Permutation Equivariant. Stability properties in two perturbation models.
- [Stability of Graph Scattering Transforms](https://arxiv.org/pdf/1906.04784.pdf)

### Theoretic analysis on GNN
- [Approximation Ratios of Graph Neural Networks for Combinatorial Problems](https://arxiv.org/pdf/1905.10261.pdf)

## Knowlede Graphs
- [Learning to Exploit Long-term Relational Dependencies in Knowledge Graphs](https://arxiv.org/pdf/1905.04914.pdf)

### Not published
#### Node Classification
- [Revisiting Graph Neural Networks: All We Have is Low-Pass Filters](https://arxiv.org/pdf/1905.09550.pdf)
- [Edge Contraction Pooling for Graph Neural Networks](https://arxiv.org/pdf/1905.10990.pdf) (Node and graph classification)
- [Variational Spectral Graph Convolutional Networks](https://arxiv.org/pdf/1906.01852.pdf)
	- Noisy graphs.
- [Dimensional Reweighting Graph Convolutional Networks](https://arxiv.org/pdf/1907.02237.pdf)

#### Graph Classification
- [Are Powerful Graph Neural Nets Necessary? A Dissection on Graph Classification](https://arxiv.org/pdf/1905.04579.pdf)
- [Neighborhood Enlargement in Graph Neural Networks](https://arxiv.org/pdf/1905.08509.pdf)
- [Provably Powerful Graph Networks](https://arxiv.org/pdf/1905.11136.pdf)
- [Wasserstein Weisfeiler-Lehman Graph Kernels](https://arxiv.org/pdf/1906.01277.pdf)
- [Graph Filtration Learning](https://arxiv.org/pdf/1905.10996.pdf)
- [Improving Attention Mechanism in Graph Neural Networks via Cardinality Preservation](https://arxiv.org/pdf/1907.02204.pdf)

#### Generative Model
- [MolecularRNN: Generating realistic molecular graphs with optimized properties](https://arxiv.org/pdf/1905.13372.pdf)
- [A Two-Step Graph Convolutional Decoder for Molecule Generation](https://arxiv.org/pdf/1906.03412.pdf)

#### Others
- [Relational Reasoning using Prior Knowledge for Visual Captioning](https://arxiv.org/pdf/1906.01290.pdf)
- [Discovering Neural Wirings](https://arxiv.org/pdf/1906.00586.pdf)
- [LEARNING REPRESENTATIONS OF GRAPH DATA: A SURVEY](https://arxiv.org/pdf/1906.02989.pdf)
- [Redundancy-Free Computation Graphs for Graph Neural Networks](https://arxiv.org/pdf/1906.03707.pdf)
- [Representation Learning on Networks: Theories, Algorithms, and Applications](https://dl.acm.org/citation.cfm?id=3320095), [[Tutorial]](http://snap.stanford.edu/proj/embeddings-www/)
- [Attributed Graph Clustering: A Deep Attentional Embedding Approach](https://arxiv.org/pdf/1906.06532.pdf)


## CNN Ideas
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
- [A Convolutional Neural Network for Modelling Sentences](https://www.aclweb.org/anthology/P14-1062)
	- [참고](https://ratsgo.github.io/deep%20learning/2017/10/09/CNNs/), [참고2](https://ratsgo.github.io/natural%20language%20processing/2017/08/16/deepNLP/)

## Multi-hop QA
I’m thinking an interesting direction for you to go is to do multi-hop reading comprehension, for datasets like DROP and HotpotQA
- DROP: https://allennlp.org/drop
- HotpotQA: https://hotpotqa.github.io/

To answer questions in these QA datasets require identifying evidences scattered in different places of the paragraphs, and reasoning across these evidences to draw the correct answer. This asks for capabilities including retrieval, coreference, reasoning, etc. This is not symbolic reasoning but more of an implicit reasoning over unstructured text.

Some relevant reads:
- https://arxiv.org/abs/1906.02900
- https://arxiv.org/abs/1904.12106
- https://arxiv.org/abs/1906.02916
- https://arxiv.org/abs/1905.06933
- https://arxiv.org/abs/1906.05210 

