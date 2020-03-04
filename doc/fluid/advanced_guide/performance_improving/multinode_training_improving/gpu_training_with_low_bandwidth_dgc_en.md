
# Distributed GPU training on low-bandwidth networks

## 1. Background
Large-scale distributed training requires significant communication bandwidth for gradient exchange that limits the scalability of multi-node training, and requires expensive high-bandwidth network infrastructure. Distributed training in environments such as low-bandwidth cloud networks becomes even worse. Existing [Deep Gradient Compression](https://arxiv.org/abs/1712.01887) research shows that 99.9% of the gradient exchanges in distributed SGD are redundant. We can use deep gradient compression to select important gradients for communication to reduce communication size and reduce dependence on communication bandwidth. Paddle has implemented DGC sparse communication methods, which can effectively perform GPU distributed training on low-bandwidth networks. The following will describe how to use DGC, its application scenarios, and basic principles.

## 2. How to use
`Note: Please use Paddle 1.6.2 or later versions when using DGC. There have some bugs in previous versions on DGC.` 
The DGC sparse communication algorithm is provided in the form of the DGCMomentumOptimizer interface. Currently, only GPU multi-card and GPU multi-machine distribution are supported.  Because the existing fuse strategy will cause DGC to fail, so you need to set `strategy.fuse_all_reduce_ops = False` to disable fuse when using DGC. DGC only supports the Momentum optimizer. When using it, replace the Momentum optimizer in the current code to DGCMomentumOptimizer, and add the parameters required by DGC. As shown in the following code, rampup\_begin\_step represents the steps that DGC start to run, more detailed parameters can be found in [api documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/api/optimizer/DGCMomentumOptimizer.html).
``` python
import paddle.fluid as fluid
# optimizer = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
# Replace the Momentum optimizer and add the parameters required by DGC 
optimizer = fluid.optimizer.DGCMomentumOptimizer(
    learning_rate=0.001, momentum=0.9, rampup_begin_step=0)
optimizer.minimize(cost)
```
An [example of DGC](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/dgc_example) is provided in fleet. The example uses digital handwriting recognition as an example. The program is first ported to a distributed version (Note: DGC also supports multi-cards), then add the DGC optimizer. You can refer to this example to migrate the single-card program to DGC. In the process of migrating a single-machine single-card program to DGC, it is generally necessary to first align the accuracy of multi-machine Momentum, and then align the accuracy of DGC.

## 3. Hyperparameter tuning & suitable scenarios
### 3.1 Warm-up training hyperparameter tuning
For pre-training, warm-up training is generally required when using DGC, otherwise, some accuracy may be lost. The following figure is the training result of the Imagenet dataset of the ResNet50 model. DGC without warm-up training eventually lost about 0.3% accuracy.
<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_guide/performance_improving/multinode_training_improving/images/dgc_resnet50_acc1.png" width="400"/>
</p>

You can refer to the settings of the paper for warm-up training hyperparameter tuning. For image classification, the paper uses 4 epochs for warm-up training on a total of 164 and 90 epochs on the Cifar10 and ImageNet datasets. On the language model PTB dataset, one epoch was selected for warm-up training in a total of 40 epochs training. On the speech recognition AN4 dataset, one epoch is selected from 80 epochs for warm-up training. 
75%, 93.75%, 98.4375%, 99.6%, and 99.9% sparsity gradually increasing strategies have been used in the paper. Since AllGather is used for paddle sparse gradient aggregation communication, the communication size will increase with the number of cards, so it is not recommended to warm-up training with lower sparsity when the number of cards is large. For example, at 75% sparsity, each card will choose a 25% gradients for communication. When the number of cards is 32, the communication size is 32\*(1-0.75)=8 times to the normal dense communication. Therefore, it is better to use normal dense communication in the first few epochs. Can refer to the following:
``` python
# 1. Take 1252 steps as an epoch, the first 2 epochs use normal dense communication, and the last 3 epochs gradually increase the sparsity to 99.9% 
optimizer = fluid.optimizer.DGCMomentumOptimizer(
    learning_rate=0.001, momentum=0.9, rampup_begin_step=1252*2,
    rampup_step=1252*3, sparsity=[0.984375, 0.996, 0.999])
# 2. The first 4 epochs use dense communication. After that, use default 0.999 sparsity 
optimizer = fluid.optimizer.DGCMomentumOptimizer(
    learning_rate=0.001, momentum=0.9, rampup_begin_step=1252*4)
```
For fine-tuning training, existing tests have shown that warm-up training is not required and DGC can be used directly from the 0th epoch.
``` python
# DGC sparse communication start from step 0
optimizer = fluid.optimizer.DGCMomentumOptimizer(
    learning_rate=0.001, momentum=0.9, rampup_begin_step=0)
```
### 3.2 Suitable scenarios
DGC sparse communication will have a large performance improvement in the case of low-bandwidth with communication bottlenecks, but in the case of single-node multi-cards and RDMA networks which communication is not a bottleneck, it will not bring about performance improvements. At the same time, due to the use of AllGather the communication size will increase with the number of cards, so the multi-machine training scale of DGC should not be too large. Therefore, DGC is suitable for low-bandwidth networks, and the scale of nodes should not be too large, such as >128 cards. When a network is on the cloud or high-bandwidth network equipment is expensive, DGC can effectively reduce training costs.

## 4. Principle
The principle of this section is basically from the [Deep Gradient Compression](https://arxiv.org/abs/1712.01887). This article extracts some parts. For a more detailed understanding, it is recommended to read the paper directly.
### 4.1 Gradient Sparsification
The basic idea of DGC is to reduce the use of communication bandwidth by sending only the important gradients, that is, only gradients larger than a threshold are transmitted. To avoid losing information, DGC accumulate the rest of the gradients locally. Eventually, these gradients become large enough to be transmitted.
The insight is that the local gradient accumulation is equivalent to increasing the batch size over time (DGC is equivalent to each gradient having its own batch size). Let $F(w)$ be the loss function that we want to optimize. Synchronous Distributed SGD performs the following update with N training nodes in total:
$$
F(w)=\\frac{1}{\|\\chi\|}\\sum\_{x\\in\\chi}f(x, w), \\qquad w\_{t+1}=w\_{t}-\\eta\\frac{1}{N b}\\sum\_{k=1}^{N}\\sum\_{x\\in\\mathcal{B}\_{k,t}}\\nabla f\\left(x, w\_{t}\\right) \\tag{1}
$$
where $\chi$ is the training dataset, $w$ is the weights of a network, $f(x, w)$ is the loss computed from samples $x \in \chi$, $\eta$ is the learning rate, N is the number of training nodes, and $\mathcal{B}\_{k, t}$ is a sequence of N mini-batches sampled from $\chi$ at node $k$ iteration $t$, each of size b.
Consider the weight value $w^{(i)}$ of i-th position in flattened weights $w$. After T iterations, we have
$$
w\_{t+T}^{(i)}=w\_{t}^{(i)}-\\eta T \\cdot \\frac{1}{N b T} \\sum\_{k=1}^{N}\\left(\\sum\_{\\tau=0}^{T-1} \\sum\_{x \\in \\mathcal{B}\_{k, t+\\tau}} \\nabla^{(i)} f\\left(x, w\_{t+\\tau}\\right)\\right)  \\tag{2}
$$
Equation 2 shows that local gradient accumulation can be considered as increasing the batch size from $Nb$ to $NbT$, where T is the length of the sparse update interval between two iterations.
### 4.2 Improving the local gradient accumulation
Without care, the sparse update will greatly harm convergence. DGC using momentum correction and local gradient clipping to mitigate this problem.
#### 4.2.1 Momentum correction
Distributed training with vanilla momentum SGD on N training nodes follows,
$$
u\_{t}=m u\_{t-1}+\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}\\right), \\quad w\_{t+1}=w\_{t}-\\eta u\_{t}  \\tag{3}
$$
where $m$ is the momentum, $N$ is the number of training nodes, and $\\nabla\_{k, t}=\\frac{1}{N b} \\sum\_{x \\in \\mathcal{B}\_{k, t}} \\nabla f\\left(x, w\_{t}\\right)$.
Consider the weight value $w^{(i)}$ of i-th position in flattened weights $w$. After T iterations, the change in weight value $w^{(i)}$ shows as follows,
$$
w\_{t+T}^{(i)}=w\_{t}^{(i)}-\\eta\\left[\\cdots+\\left(\\sum\_{\\tau=0}^{T-2} m^{\\tau}\\right) \\nabla\_{k, t+1}^{(i)}+\\left(\\sum\_{\\tau=0}^{T-1} m^{\\tau}\\right) \\nabla\_{k, t}^{(i)}\\right]  \\tag{4}
$$
If SGD with the momentum is directly applied to the sparse gradient scenario, then update rule becomes:
$$
v\_{k, t}=v\_{k, t-1}+\\nabla\_{k, t}, \\quad u\_{t}=m u\_{t-1}+\\sum\_{k=1}^{N} \\operatorname{sparse}\\left(v\_{k, t}\\right), \\quad w\_{t+1}=w\_{t}-\\eta u\_{t} \\tag{5}
$$
where $v\_k$ is the local gradient accumulation on the training node k. Once the accumulation results $v\_k$ is larger than a threshold, it will be encoded and get sent over the network in the second term and gets cleared by the mask in the sparse() function.
The change in weight value $w^{(i)}$ after the sparse update interval T becomes,
$$
w\_{t+T}^{(i)}=w\_{t}^{(i)}-\\eta\\left(\\cdots+\\nabla\_{k, t+1}^{(i)}+\\nabla\_{k, t}^{(i)}\\right) \\tag{6}
$$
The disappearance of the accumulated discounting factor $\sum\_{\tau=0}^{T-1} m^{\tau}$ in Equation 6 compared to Equation 4 leads to the loss of convergence performance. It is illustrated in Figure (a), where Equation 4 drives the optimization from point A to point B, but with local gradient accumulation, Equation 4 goes to point C. When the gradient sparsity is high, the significant side effect will harm the model performance. To avoid this error, we need momentum correction on top of Equation 5 to make sure the sparse update is equivalent to the dense update as in Equation 3.
<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_guide/performance_improving/multinode_training_improving/images/dgc_without_momentum_correction.png" width="320"/>
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_guide/performance_improving/multinode_training_improving/images/dgc_with_momentum_correction.png" width="320"/>
</p>

If we regard the velocity $u\_t$ in Equation 3 as "gradient", the second term of Equation 3 can be considered as the vanilla SGD for the "gradient" $u\_t$. The local gradient accumulation is proved to be effective for the vanilla SGD in Section 4.1. Therefore, we can locally accumulate the velocity $u\_t$ instead of the real gradient $\nabla\_{k, t}$ to migrate Equation 5 to approach Equation 3:
$$
u\_{k, t}=m u\_{k, t-1}+\\nabla\_{k, t}, \\quad v\_{k, t}=v\_{k, t-1}+u\_{k, t}, \\quad w\_{t+1}=w\_{t}-\\eta \\sum\_{k=1}^{N} \\operatorname{sparse}\\left(v\_{k, t}\\right)  \\tag{7}
$$
After correction, as shown in (b) above, the equation can be normally changed from A Point to point B. Beyond the vanilla momentum SGD, the paper also gives the correction equation of Nesterov momentum SGD.
#### 4.2.2 Local gradient clipping
Gradient clipping is widely adopted to avoid the exploding gradient problem. The method proposed by Pascanu et al. (2013) rescales the gradients whenever the sum of their L2-norms exceeds a threshold. This step is conventionally executed after gradient aggregation from all nodes. Because DGC accumulates gradients over iterations on each node independently, so DGC performs the gradient clipping locally before adding the current gradient $G\_t$ to previous accumulation. DGC scale the threshold by $N^{-1/2}$,
$$
thr\_{G^{k}}=N^{-1 / 2} \\cdot thr\_{G}  \\tag{8}
$$
### 4.3 Overcoming the staleness effect
Because we delay the update of small gradients, when these updates do occur, they are outdated or stale. Most of the parameters are updated every 600 to 1000 iterations when gradient sparsity is 99.9%. Staleness can slow down convergence and degrade model performance. DGC mitigate staleness with momentum factor masking and warm-up training.
#### 4.3.1 Momentum factor masking
DGC introduce momentum factor masking to alleviate staleness:
$$
Mask \\leftarrow\\left|v\_{k, t}\\right|>t h r, \\quad v\_{k, t} \\leftarrow v\_{k, t} \\odot \\neg Mask, \\quad u\_{k, t} \\leftarrow u\_{k, t} \\odot \\neg Mask \\tag{9}
$$
This mask stops the momentum for delayed gradients, preventing the stale momentum from carrying the weights in the wrong direction.

### 4.3.2 Warm-up training
In the early stages of training, the gradient is changing sharply, the stale gradient will have a large impact, so the weights need to be updated in time. Therefore, DGC adopts the warm-up training method. During the warm-up period, a smaller learning rate is used to slow down the changing speed of the network, and a smaller sparsity is used to reduce the number of gradients being delayed. During the warm-up period of DGC, the learning rate increases linearly and the gradient sparsity exponentially increases to the final value.

### 4.4 Regularization (Weight Decay) Correction
Paddle framework implements regularization in the form of Weight Decay.  Taking L2Decay as an example, after adding weight decay to the vanilla momentum SGD on Equation 3, the formula become:
$$
G\_{t}=\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}\\right)+\\lambda w\_{t}, \\quad  u\_{t}=m u\_{t-1}+G\_{t}, \\quad w\_{t+1}=w\_{t}-\\eta u\_{t} \\tag{10}
$$
where $\lambda$ is the Weight Decay coefficient, and $G\_{t}$ is the aggregation gradient after adding the L2Decay term. Since the local momentum correction is performed in Equation 7. According to the same idea, we apply the corrected Weight Decay on the local gradient. As the following formula, add the local Weight Decay term to the local gradient.
$$
\\nabla\_{k, t}=\\nabla\_{k, t}+\\frac{\\lambda}{N} w\_{t} \\tag{11}
$$
In actual training, the coefficient of weight decay is usually set to $\lambda=10^{-4}$. When there are many cards, such as 32 cards, the local weight decay coefficient will be $\frac{\lambda}{N}=\frac{10^{-4}}{32}=3.125\*10^{-6}$, which is low in numerical accuracy, and an accuracy loss is found during the training. Therefore, a numerical correction is needed for the local weight decay. As the following formula,
$$
\\nabla\_{k, t}^{'}=N \\nabla\_{k, t}+\\lambda w\_{t}, \\quad
G\_{t}^{'}=\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}^{'}\\right)=N\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}\\right)+N\\lambda w\_{t}, \\quad
G\_{t}=\\frac{G\_{t}^{'}}{N}=\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}\\right)+\\lambda w\_{t} \\tag{12}
$$
The specific method is to multiply the local gradient by the number of cards to obtain $\nabla\_{k, t}^{'}$. At this time, the $\lambda$ no need to be divided by the number of cards. Aggregate the gradient to get $G\_{t}^{'}$, then divide the aggregation gradient by the number of cards to get the required $G\_{t}$.
