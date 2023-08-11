# Violin: Virtual Overbridge Linking

(Still under construction...)

<p align="center"><img alt='The pipeline of Violin.' width="100%" src="assets/fig_violin_pipeline.png" /></p>

This is the code repo for the **IJCAI 2023** paper "Violin: Virtual Overbridge Linking for Enhancing Semi-supervised Learning on Graphs with Limited Labels".
You can also find a preliminary version of the paper and the appendix in the [./appendix](./appendix) directory.

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


## Cite This Work
```
@article{xie2023violin,
  title={Violin: Virtual Overbridge Linking for Enhancing Semi-supervised Learning on Graphs with Limited Labels},
  author={Xie, Siyue and Tam, Da Sun Handason and Lau, Wing Cheong},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI},
  year={2023},
}
```


## Requirements
- Numpy >= 1.23.1
- PyTorch >= 1.9.0
- PyTorch Geometric == 2.0.4
- scikit-learn >= 1.1.2
- ogb == 1.2.1


> Note: the code may not be compatible with the latest PyTorch Geometric and ogb libraries.
Please modify the code accordingly if you encounter any errors (especially the part of loading datasets).


## Instructions

### Quick Start

Two steps to quickly reproduce the results on the Cora, Citeseer, Pubmed and Ogbn-arxiv datasets, take GCN as the backbone as an example:
1. Train the vanilla GCN. Type the following commands:
   - Cora: ```python main.py --model GCN --dataset Cora```
   - Citeseer: ```python main.py --model GCN --dataset Citeseer```
   - Pubmed: ```python main.py --model GCN --dataset Pubmed```
   - Ogbn-arxiv: ```python main.py --model GCN --dataset ogbn-arxiv --n_layers 3 --hid_dim 256 --bn --dropout 0.5```
2. Enhance the trained GCN model using Violin. Type the following command:
   - Cora: ```python main.py --model ViolinGCN --dataset Cora --alpha 0.8 --delta 0.9```
   - Citeseer: ```python main.py --model ViolinGCN --dataset Citeseer --alpha 0.2 --delta 0.8```
   - Pubmed: ```python main.py --model ViolinGCN --dataset Pubmed --alpha 0.4 --delta 0.9 --cls_mode both --m 2```
   - Ogbn-arxiv: ```python main.py --model ViolinGCN --dataset ogbn-arxiv --n_epochs 500 --n_layers 3 --hid_dim 256 --gamma 0.4 --alpha 0.6 --delta 0.8 --cls_mode both --m 2 --dropout 0.5 --lr 0.005 --bn```

If you would like to speed up the training with a GPU, you can specify the ID of your GPU card by adding `--gpu [gpu_id]` to the command, for example:
1. Train the vanilla GCN, with GPU#0. Type the following commands:
   - Cora: ```python main.py --model GCN --dataset Cora --gpu 0```
2. Enhance the trained GCN model using Violin, with GPU#0. Type the following command:
   - Cora:```python main.py --model ViolinGCN --dataset Cora --alpha 0.8 --delta 0.9 --gpu 0```

### Datasets

We conduct experiments on several experiments, including Cora, Citeseer, Pubmed and Ogbn-arxiv.
These four datasets are all paper citation networks, i.e., a node stands for a paper and edge between two nodes indicate a reference/ citation relation between two paper.
The train-validation-test split follows the standard/ official splits of the corresponding dataset.
The data statistics are provided as follows:

| Dataset    | \# Nodes  | \# Edges  | \# Classes    | \# Attributes | Train/ Val/ Test Split    | Label Rate (\%)   |
| ---------  | --------- | --------  | ------------- | ------------- | ------------------------- | ----------------- |
| Cora       | 2708      | 5228      | 7             | 1433          | 140/ 500/ 1000            | 5.17              |
| Citeseer   | 3327      | 4614      | 6             | 3703          | 120/ 500/ 1000            | 3.62              |
| Pubmed     | 19717     | 44338     | 3             | 500           | 60/ 500/ 1000             | 0.30              |
| Ogbn-arxiv | 169343    | 2315598   | 40            | 128           | 90941/ 29799/ 48603       | 53.70             |

We also conduct experiments on Amazon-Photos and Coauthor-CS datasets.
Results on these additional datasets refer to our [appendix](./appendix/IJCAI2023_Violin_appendix.pdf).

### Explanations of the Arguments/ Hyper-parameters
You can modify the training settings by using different hyper-parameters.
Settings corresponding to the reported results are detailed in Table 6 of Appendix D.3.

Here we provide explanations of all tunable arguments/ hyper-parameters in the `main.py` file:

- round: the number of rounds of training. Default: 10. 
> Note: the reported results are the averaged test accuracy of 10 rounds of training.

- model: specify the name of the model. Default: `GCN`. 
> Note: the model name should be consistent with the name of the corresponding model class in `models.py` or `models_ogb.py`. If you want to enhance a backbone model, please make sure that you have already had a trained backbone model for Violin to start with.

- dataset: specify the name of the dataset. Default: `Cora`. Options: `Cora`, `Citeseer`, `Pubmed`, `ogbn-arxiv`.

- n_epochs: the number of training epochs. Default: 200.

- eta: the gap of epochs to update the estimated labels, only available for Violin. Default: 1.

- n_layers: the number of layers in the backbone model. Default: 2.

- hid_dim: the hidden dimension of the backbone model. Default: 16.

- dropout: the dropout rate of the backbone model. Default: 0.6.

- gpu: the ID of the GPU card. Default: -1 (CPU).

- lr: the learning rate. Default: 0.01.

- weight_decay: the weight decay. Default: 5e-4.

- alpha: the weight of the consistency loss, only available for Violin. Please refer to Equ.(9) of our paper. Default: 0.8.

- gamma: the weight of the VO loss, only available for Violin. Please refer to Equ.(9) of our paper. Default: 0.4.

- delta: the accuracy requirement, only available for Violin. Please refer to Equ.(5) of our paper. Default: 0.9.

- cls_mode: the mode of the supervised classification loss, only available for Violin. Please refer to Equ.(10) of our paper. Default: `virt`. Options: `ori` (only use the 1st term of Equ.(10)), `virt` (only use the 2nd term of Equ.(10)), `both` (use both terms, i.e., exactly the same form of Equ.(10)).

- m: the number of virtual overbridges to be added between two nodes, only available for Violin. Default: 1.

- bn: whether to use batch normalization. Default: False.


## Experimental Results

```
n2v: node2vec
SAGE: GraphSAGE
```


| Models        | Cora              | Citeseer        | Pubmed          | Ogbn-arxiv            |
| ---------     | -----             | --------        | -------         | -------------         |
| ViolinGCN_16  | **85.22** &pm; 0.60   | 73.38 &pm; 0.32 | 81.11 &pm; 0.47 | -                 | 
| ViolinGCN_32  | 85.08 &pm; 0.65   | 73.96 &pm; 0.38 | 81.05 &pm; 0.51 | -                     | 
| ViolinGCN_128 | 84.49 &pm; 0.66   | **74.26** &pm; 0.40 | **81.23** &pm; 0.42 | -             | 
| ViolinGCN_256 | 84.03 &pm; 0.59   | 74.16 &pm; 0.55 | 80.83 &pm; 0.36 | **72.49** &pm; 0.09   | 
| ---           | ---               | ---             | ---             | ---                   |
| MLP           | 58.51 &pm; 0.80   | 55.64 &pm; 0.46 | 72.71 &pm; 0.61 | 55.50 &pm; 0.23       |
| n2v           | 72.35 &pm; 1.41   | 50.82 &pm; 0.96 | 62.03 &pm; 1.05 | 70.07 &pm; 0.13       |
| DGI           | 82.30 &pm; 0.60   | 71.80 &pm; 0.70 | 76.80 &pm; 0.60 | N/A                   |
| MVGRL         | 82.90 &pm; 0.70   | 72.60 &pm; 0.70 | 79.40 &pm; 0.30 | N/A                   |  
| DIMP          | 83.30 &pm; 0.50   | 73.30 &pm; 0.50 | 81.40 &pm; 0.50 | N/A                   | 
| PNA           | 75.05 &pm; 3.37   | 55.04 &pm; 4.85 | 75.91 &pm; 1.17 | 69.54 &pm; 0.58       | 
| GIN           | 78.83 &pm; 1.45   | 66.87 &pm; 0.96 | 77.83 &pm; 0.42 | 63.19 &pm; 1.57       | 
| JK-Net        | 80.35 &pm; 0.58   | 67.29 &pm; 1.02 | 78.36 &pm; 0.31 | 72.19 &pm; 0.24       | 
| SGC           | 80.70 &pm; 0.55   | 71.94 &pm; 0.07 | 78.82 &pm; 0.04 | 68.59 &pm; 0.03       | 
| SAGE          | 81.73 &pm; 0.58   | 69.86 &pm; 0.62 | 77.20 &pm; 0.40 | 72.04 &pm; 0.20       | 
| GCN           | 82.52 &pm; 0.60   | 71.02 &pm; 0.83 | 79.16 &pm; 0.35 | 71.99 &pm; 0.22       | 
| GAT           | 82.76 &pm; 0.88   | 71.87 &pm; 0.53 | 77.74 &pm; 0.34 | 71.75 &pm; 0.28       | 
| APPNP         | 83.13 &pm; 0.58   | 71.39 &pm; 0.68 | 80.30 &pm; 0.17 | 71.22 &pm; 0.26       | 
| GCNII         | 84.17 &pm; 0.40   | 72.46 &pm; 0.74 | 79.85 &pm; 0.34 | 72.46 &pm; 0.32       | 
| AdaEdge       | 82.30 &pm; 0.80   | 69.10 &pm; 0.90 | 77.40 &pm; 0.50 | N/A                   | 
| CG^3          | 83.40 &pm; 0.70   | 73.60 &pm; 0.80 | 80.20 &pm; 0.80 | N/A                   | 
| CAUG          | 83.60 &pm; 0.50   | 73.30 &pm; 1.10 | 79.30 &pm; 0.40 | 71.40 &pm; 0.50       | 
| CoCoS         | 84.15 &pm; N/A    | 73.57 &pm; N/A  | 80.92 &pm; N/A  | 71.77 &pm; N/A        | 
| GRAND         | 84.50 &pm; 0.30   | 74.20 &pm; 0.30 | 80.00 &pm; 0.30 | N/A                   | 
| GAM           | 84.80 &pm; 0.06   | 72.46 &pm; 0.44 | 81.00 &pm; 0.09 | N/A                   | 

For more detailed ablation studies and additional experiments, please refer to our paper and the corresponding 
supplementary materials.