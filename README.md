# How Powerful are Graph Neural Networks?

This repository is the official PyTorch implementation of the experiments in the following paper: 

Keyulu Xu*, Weihua Hu*, Jure Leskovec, Stefanie Jegelka. How Powerful are Graph Neural Networks? ICLR 2019. 

[arXiv](https://arxiv.org/abs/1810.00826) [OpenReview](https://openreview.net/forum?id=ryGs6iA5Km) 

If you make use of the code/experiment or GIN algorithm in your work, please cite our paper (Bibtex below).
```
@inproceedings{
xu2018how,
title={How Powerful are Graph Neural Networks?},
author={Keyulu Xu and Weihua Hu and Jure Leskovec and Stefanie Jegelka},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=ryGs6iA5Km},
}
```

## Installation
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 0.4.1 and 1.0.0 versions.

Then install the other dependencies.
```
pip install -r requirements.txt
```

## Test run
Unzip the dataset file
```
unzip dataset.zip
```

and run

```
python main.py
```

The default parameters are not the best performing-hyper-parameters used to reproduce our results in the paper. Hyper-parameters need to be specified through the commandline arguments. Please refer to our paper for the details of how we set the hyper-parameters. For instance, for the COLLAB and IMDB datasets, you need to add `--degree_as_tag` so that the node degrees are used for input node features.

To learn hyper-parameters to be specified, please type
```
python main.py --help
```



## Cross-validation strategy in the paper
The cross-validation in our paper only uses training and validation sets (no test set) due to small dataset size. Specifically, after obtaining 10 validation curves corresponding to 10 folds, we first took average of validation curves across the 10 folds (thus, we obtain an averaged validation curve), and then selected a single epoch that achieved the maximum averaged validation accuracy. Finally, the standard devision over the 10 folds was computed at the selected epoch. 

## Create custom dataset
If you have a set of labeled graphs you can transform it into a single .txt file which then can be fed to the network. Structure of the .txt is following:
- each graph is a block
- first line of a block consist of *%number of nodes%* *%class label%*
- each following line describes single node in a way: *%node label%* *%number of connected nodes%* *%connected node #1%* *%connected node #2%* *%connected node #3%*...
- row number correspond to the node's index, starting from 0
- test/train partition is defined by cross-validation and doesn't appear in .txt

For example:
```
10 7
0 3 1 2 9
0 3 0 2 9
0 4 0 1 3 9
0 3 2 4 5
0 3 3 5 6
0 5 3 4 6 7 8
0 4 4 5 7 8
0 3 5 6 8
0 3 5 6 7
1 3 0 1 2
```
The block corespond to graph which consist of 10 nodes and belongs to class 7. First (0) node has label 0 and has 3 neighbours; these neighbours are nodes 1, 2, and 9. The same can be applied to the next nodes.
