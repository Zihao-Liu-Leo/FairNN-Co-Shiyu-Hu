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

1.Caliskan, Aylin, Joanna J. Bryson, and Arvind Narayanan: Semantics derived automatically from language corpora contain human-like biases. Science 356.6334, 183--186 (2017)
 
2.Moritz Hardt, Eric Price, and Nathan Srebro: Equality of Opportunity in Supervised Learning. Advances in Neural Information Processing Systems 29, (2016)
 
3.Elias Benussi, Andrea Patane, Matthew Wicker, Luca Laurenti, and Marta Kwiatkowska: Individual Fairness Guarantees for Neural Networks. International Joint Conference on Artificial Intelligence Main Track 31, 651--658 (2022)
 
4.Suresh Harini, and John Guttag: A framework for understanding sources of harm throughout the machine learning life cycle. Equity and access in algorithms, mechanisms, and optimization, 1--9 (2021)

5.Olteanu Alexandra, Emre Kıcıman, and Carlos Castillo: A critical review of online social data: Biases, methodological pitfalls, and ethical boundaries. Proceedings of the eleventh ACM international conference on web search and data mining, (2018)

6.Mengnan Du, Fan Yang, Na Zou, and Xia Hu: Fairness in deep learning: A computational perspective. IEEE Intelligent Systems 36.4, 25--34 (2020)

7.Burnaev Evgeny, Pavel Erofeev, and Artem Papanov: Influence of resampling on accuracy of imbalanced classification. Eighth international conference on machine vision Vol. 9875. SPIE, (2015)

8.Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie:Class-balanced loss based on effective number of samples. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, (2019)

9.NV Chawla, KW Bowyer, LO Hall, and WP Kegelmeyer: SMOTE: synthetic minority over-sampling technique. Journal of artificial intelligence research 16, 321--357(2002) 

10.Shorten Connor, and Taghi M. Khoshgoftaar: A survey on image data augmentation for deep learning. Journal of big data 6.1, 1--48(2019) 

11.John Philips George, Deepak Vijaykeerthy, and Diptikalyan Saha: Verifying individual fairness in machine learning models. Conference on Uncertainty in Artificial Intelligence. PMLR, (2020)
