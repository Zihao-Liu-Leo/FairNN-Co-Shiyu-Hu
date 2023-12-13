# Fair-NN experiments

by Liu Zihao, Hu Shiyu

## Introduction

This project aims to implement the validation and optimization of a neural network fairness guarantee framework. We trained a neural network with adjustable independent variable parameters w and b based on the Sigmoid function as the activation function on the Boston dataset with bias attributes, and simply implemented the measure of individual fairness mentioned in the reference paper, and also made valuable optimization of fairness guarantee by applying the joint method of parameter random sampling + sample resampling + proposed gradient descent. The specific experimental code (integrated version) is shown in "fairnn.py", and the distribution code during the completion of the experiment will be attached to the "Preliminary Work" folder.


## Requirements
```
pip install argparse tqdm numpy scipy pandas matplotlib sklearn pulp
```
## Usage
```
python fairnn.py --resample 1
```
You may change the random seed using the --seed argument, extend the Sim-Grad iter using --epochs, change the learning rate in Sim-Grad using --lr, and decide whether to apply resample using --resample.

## Citation

pass
