# Violin: Virtual Overbridge Linking

(Still under construction...)

<p align="center"><img alt='The pipeline of Violin.' width="100%" src="assets/fig_violin_pipeline.png" /></p>

This is the code repo for the **IJCAI 2023** paper "Violin: Virtual Overbridge Linking for Enhancing Semi-supervised Learning on Graphs with Limited Labels".
You can also find the appendix of Violin in the [./appendix](./appendix) directory.

Violin is a follow-up work of [CoCoS](https://github.com/xslangley/cocos). 
Both of them are designed to enhance the performance of GNNs in semi-supervised node classification tasks when labels available for training are limited.

The basic information of the is as provided as follows:

> Title: Violin: Virtual Overbridge Linking for Enhancing Semi-supervised Learning on Graphs with Limited Labels

> Authors: Siyue Xie, Da Sun Handason Tam and Wing Cheong Lau

> Affiliation: The Chinese University of Hong Kong

> Abstract:  Graph Neural Networks (GNNs) is a family of promising tools for graph semi-supervised learning. 
However, in training, most existing GNNs rely heavily on a large amount of labeled data,
which is rare in real-world scenarios. 
Unlabeled data with useful information are usually under-exploited, which limits the representation power of GNNs. 
To handle these problems, we propose Virtual Overbridge Linking (Violin), a generic framework to enhance the learning capacity of common GNNs. 
By learning to add virtual overbridges between two nodes that are estimated to be
semantic-consistent, labeled and unlabeled data can be correlated. Supervised information can be well utilized in training while simultaneously inducing the model to learn from unlabeled data. 
Discriminative relation patterns extracted from unlabeled nodes can also be shared with other nodes even if they are remote from each other. 
Motivated by recent advances in data augmentations, we additionally integrate Violin with the consistency regularized training. 
Such a scheme yields node representations with better robustness, which significantly enhances a GNN. 
Violin can be readily extended to a wide range of GNNs without introducing additional learnable parameters. 
Extensive experiments on six datasets demonstrate that our method is effective and robust under low-label rate scenarios, where Violin can boost some GNNs’ performance by over 10% on node classifications.




## Requirements
- Numpy >= 1.23.1
- PyTorch >= 1.9.0
- PyTorch Geometric == 2.0.4
- scikit-learn >= 1.1.2
- ogb == 1.2.1


> Note: the code may not be compatible with the latest PyTorch Geometric and ogb libraries.
Please modify the code accordingly if you encounter any errors (especially the part of loading dataset).


## Instructions
Two steps to quickly reproduce the results on the Cora, Citeseer, Pubmed and Ogbn-arxiv datasets:
1. train the vanilla GCN. Type the following commands:
   - Cora: ```python main.py --model GCN --dataset Cora```
   - Citeseer: ```python main.py --model GCN --dataset Citeseer```
   - Pubmed: ```python main.py --model GCN --dataset Pubmed``
   - Ogbn-arxiv: ```python main.py --model GCN --dataset ogbn-arxiv --n_layers 3 --n_hidden 256 --bn --dropout 0.5``
2. enhance the trained GCN model using Violin. Type the following command:
   - Cora: ```python main.py --model ViolinGCN --dataset Cora --alpha 0.8 --delta 0.9```
   - Citeseer: ```python main.py --model ViolinGCN --dataset Citeseer --alpha 0.2 --delta 0.8```
   - Pubmed: ```python main.py --model ViolinGCN --dataset Pubmed --alpha 0.4 --delta 0.9 --cls_mode both --m 2```
   - Ogbn-arxiv: ```python main.py --model ViolinGCN --dataset ogbn-arxiv --n_layers 3 --hid_dim 256 --bn --dropout 0.5 --gamma 0.4 --alpha 0.6 --delta 0.8 --cls_mode both --m 2```

If you would like to speed up the training with a GPU, you can specify the ID of your GPU card by adding `--gpu [gpu_id]` to the command, e.g.,:
1. train the vanilla GCN, with GPU#0. Type the following commands:
   - Cora: ```python main.py --model GCN --dataset Cora --gpu 0```
2. enhance the trained GCN model using Violin, with GPU#0. Type the following command:
   - Cora:```python main.py --model ViolinGCN --dataset Cora --alpha 0.8 --delta 0.9 --gpu 0```

You can modify the training settings by using different hyper-parameters.
Settings corresponding to the reported results are detailed in Table 6 of Appendix D.3.

