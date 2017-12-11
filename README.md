# cnn_incremental-learning
Reproductibility Challenge for ICLR 2018: Incremental Learning through Deep Adaption

**cnn_v3: Dataset CIFAR10 trained with control module based on a regular 2-layer convolutional neural network.  
cnn_bgg: Dataset CIFAR10 trained with control module based **

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

Problem: cnn_vgg changed the regular

**Partial results**
Iteration: 100. Loss: 2.1288864612579346. Accuracy: 31.85.
Iteration: 200. Loss: 1.761412501335144. Accuracy: 31.19.
Iteration: 300. Loss: 1.2932252883911133. Accuracy: 48.96.
Iteration: 400. Loss: 1.426600694656372. Accuracy: 50.98.
Iteration: 500. Loss: 1.5066254138946533. Accuracy: 52.86.
Iteration: 600. Loss: 1.0674883127212524. Accuracy: 55.4.
Iteration: 700. Loss: 1.2660120725631714. Accuracy: 57.12.
Iteration: 800. Loss: 1.0268568992614746. Accuracy: 58.46.
Iteration: 900. Loss: 0.932781457901001. Accuracy: 62.76.
Iteration: 1000. Loss: 0.8914451003074646. Accuracy: 63.41.
Iteration: 1100. Loss: 0.8505222797393799. Accuracy: 66.0.
Iteration: 1200. Loss: 1.0242605209350586. Accuracy: 64.98.
Iteration: 1300. Loss: 0.9178219437599182. Accuracy: 67.42.
Iteration: 1400. Loss: 0.8666901588439941. Accuracy: 68.61.
Iteration: 1500. Loss: 0.892946720123291. Accuracy: 68.54.
Iteration: 1600. Loss: 0.8638588786125183. Accuracy: 68.35.
Iteration: 1700. Loss: 0.8628944158554077. Accuracy: 69.15.
Iteration: 1800. Loss: 1.0401527881622314. Accuracy: 68.27.
Iteration: 1900. Loss: 0.7258371710777283. Accuracy: 70.49.
Iteration: 2000. Loss: 0.6357018947601318. Accuracy: 70.98.
Iteration: 2100. Loss: 0.44135037064552307. Accuracy: 72.99.
Iteration: 2200. Loss: 0.7241382598876953. Accuracy: 71.54.
Iteration: 2300. Loss: 0.5983885526657104. Accuracy: 71.21.
Iteration: 2400. Loss: 0.3832913935184479. Accuracy: 73.9.
Iteration: 2500. Loss: 0.5777073502540588. Accuracy: 72.51.


