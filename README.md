# AdaSparse

This repository contains the implementation for **Adaptive Sparsification for Communication-Efficient Distributed Learning** published in MobiHoc'25.


In order to use the code, please follow these steps:

## 1- Install requirements

pip install -r requirements.txt


## 2- To reproduce the results in each section of the paper, run the follwoing: 


### For Section 7.1.1:

`python main.py -learning_rate 0.5 -batch_size 500 -myseed 0 -num_epoch 50 -dataset 'MNIST' -datasplit 'iid' -model_name 'LogisticRegression' -method 'LCAdaSparse' -kappa 50.0 -epsilon 2.0 -delay_dist 'uniform' -alpha 0.0`


### For Section 7.1.2:

`python main.py -learning_rate 1e-4 -batch_size 50 -myseed 0 -num_epoch 60 -dataset 'CIFAR10' -datasplit 'non-iid' -model_name 'ResNet9' -method 'LCAdaSparse' -kappa 10.0 -epsilon 15.0 -delay_dist 'uniform' -alpha 0.0`

### For Section 7.1.3:

`python main.py -learning_rate 1e-4 -batch_size 50 -myseed 0 -num_epoch 60 -dataset 'ImageNette' -datasplit 'iid' -model_name 'ResNet9' -method 'LCAdaSparse' -kappa 2000.0 -epsilon 15.0 -delay_dist 'uniform' -alpha 0.0`


### For Section 7.2:

`python main.py -learning_rate 1e-2 -batch_size 50 -myseed 0 -num_epoch 45 -dataset 'CIFAR10' -datasplit 'iid' -model_name 'ResNet20' -method 'LCAdaSparse' -kappa 2.0 -epsilon 2.0 -delay_dist 'non-uniform' -alpha 0.5`


## Key Notes:

1. The random seed for data distribution among devices, batch sampling, and random delay generation is specified using the `-myseed` argument.

2. The `-dataset` argument can be set to `'MNIST'`, `'CIFAR10'`, or `'ImageNette'`.

3. The `-datasplit` argument can be set to `'iid'`or `'non-iid'`.

4. The `-model_name` argument can be set to `'LogisticRegression'`, `'ResNet20'`, or `'ResNet9'`.

5. To generate results for different methods, set the `-method` argument to one of the following: `'topk'`, `'varreduced'`, `'UniSparse'`, `'AdaSparse'`, or `'LCAdaSparse'`.

6. The `-kappa` argument corresponds to:
   - $k$ for the `topk` method.
   - $\kappa$ for the `varreduced` method.
   - $\kappa$ for the `AdaSparse` and `LCAdaSparse` methods, where $V = \frac{\kappa T}{n d_{\max}}$.
   - the threshold value for the `UniSparse` method.

7. The `-epsilon` argument is the $\epsilon$ parameter for the `AdaSparse` and `LCAdaSparse` methods and has no effect on the other methods.

8. The `-delay_dist` argument can be set to `'uniform'`or `'non-uniform'`.

9. The `-alpha` argument can be set to any value between 0 and 1 when `delay_dist` is set to non-uniform; otherwise, it has no effect on the result.





Thank you for your attention!



 
