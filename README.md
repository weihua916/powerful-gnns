# How Powerful are Graph Neural Networks?

This repository is the official PyTorch implementation of the experiments in the following paper: 

Keyulu Xu*, Weihua Hu*, Jure Leskovec, Stefanie Jegelka. How Powerful are Graph Neural Networks? ICLR 2019. [OpenReview](https://openreview.net/forum?id=ryGs6iA5Km)

## Installation
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 0.4.1 and 1.0.0 versions.

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

Default parameters are not the best performing-hyper-parameters. Hyper-parameters need to be specified through the commandline arguments. Type

```
python main.py --help
```

to learn hyper-parameters to be specified.

