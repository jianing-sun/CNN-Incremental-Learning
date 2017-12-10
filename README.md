# cnn_incremental-learning
Reproductibility Challenge for ICLR 2017: Incremental Learning through Deep Adaption

cnn_v3 is the right one

Based on pytorch

A project from ECSE608: Machine Learning

Due on Dec 21

12/09/2017:
Finally apply the control module with the original module with no error.

Problem: the accuracy didn't change no matter how many iterations. 
It seems the gradient descent didn't work. Still need to find the reason. 
But at least it didn't have error... a little relieved. 

Solved：change newmodel = copy.deepcopy(model) to newmodel = model

Problem: which one is better with CIFAR10? old one or the new one?
