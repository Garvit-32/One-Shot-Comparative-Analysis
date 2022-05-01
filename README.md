# Comparative Analysis of One-Shot Learning

The aim for this repository is to contain clean, readable and tested code to reproduce one-shot learning research.

This project is written in python 3.7 and Pytorch and assumes you have a GPU.

We implemented the following few shot models:

- [Prototypical Networks](https://arxiv.org/abs/1703.05175)
- [Siamese Networks](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [Matching Networks](https://arxiv.org/abs/1606.04080)
- [Model Agnostic Meta Learning ( MAML )](https://arxiv.org/abs/1703.03400)

### Data

We have implemented our models on 2 datasets, Omniglot and Mini-ImageNet

**Omniglot** dataset. Download from [Link](https://github.com/brendenlake/omniglot/tree/master/python) place the extracted files into `DATA_PATH/Omniglot_Raw` and run`scripts/preprocessing_omniglot.py`

**Mini-ImageNet** dataset. Download files from [Link](https://drive.google.com/file/d/1-31FtYmm42a1MbU67weeP1Juh7SgGq7e/view?usp=sharing)
place in `DATA_PATH/miniImageNet/images` and run`preprocessing_miniImageNet.py`

## Steps to run locally

- Clone this repository and launch code:

```
git clone https://github.com/Garvit-32/One-Shot-Comparative-Analysis
cd One-Shot-Comparative-Analysis
```

- Use pip to install other dependencies from `requirements.txt`

```
pip install -r requirements.txt
```

- Set the DATA_PATH in config.py

```
pip install -r requirements.txt
```

- Download the dataset

### Prototypical Network

- Train Network

```
python protonets.py --dataset omniglot/miniImageNet --n-train 1 --n-test 1 --k-train 5 --k-test 5
```

- Test Network

```
python test_proto.py --dataset omniglot/miniImageNet --n 1 --k 5 --path <path f the model file>
```

### Matching Network

- Train Network

```
python matching_networks.py --dataset omniglot/miniImageNet --n-train 1 --n-test 1 --k-train 5 --k-test 5
```

- Test Network

```
python test_matching.py --dataset omniglot/miniImageNet --n 1 --k 5 --q 1 --path <path f the model file>
```

### MAML

- Train Network

```
python maml.py --dataset omniglot/miniImageNet --n 1 --k 5 --q 1
```

- Test Network

```
python test_maml.py --dataset omniglot/miniImageNet --n 1 --k 5 --q 1 --path <path f the model file>
```

### Siamese

- Unzip Siamese.rar and go into directory and install the dependencies

```
cd Siamese
pip install -r requirements.txt
```

- Download Dataset

```
python main.py download-data
```

- Train network

```
python main.py train_model
```

- Test network

```
python main.py test_model
```

## Team

- [Garvit Garg](https://github.com/Garvit-32)
- [Kshitiz ](https://github.com/kshitiz-1225)
- [Mohan Chhabaria ](https://github.com/MohanChhabaria)
