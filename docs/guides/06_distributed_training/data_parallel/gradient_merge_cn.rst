.. _gradient_merge:

Gradient Merge
========================

一、简介
----------------------

为了提升模型的性能，人们开始追求：更大规模的数据集、更深的网络层、更庞大的参数规模。但是随之而来的就是给模型训练带来了巨大的压力，因此分布式技术及定制化 AI 芯片应运而生。但在分布式训练中，经常会遇到显存或者内存不足的情况，通常是以下几点原因导致的：

-  输入的数据过大，例如视频类训练数据。
-  深度模型的参数过多或过大，所需的存储空间超出了内存/显存的大小。
-  AI 芯片的内存有限。

为了能正常完成训练，我们通常只能使用较小的 batch
size 以降低模型训练中的所需要的存储空间，这将导致很多模型无法通过提高训练时的 batch
size 来提高模型的精度。

Gradient Merge (GM) 策略的主要思想是将连续多个 batch 数据训练得到的参数梯度合并做一次更新。
在该训练策略下，虽然从形式上看依然是小 batch 规模的数据在训练，但是效果上可以达到多个小 batch 数据合并成大 batch 后训练的效果。


二、原理介绍
-------------------------

Gradient Merge 只是在训练流程上做了一些微调，达到模拟出大 batch
size 训练效果的目的。具体来说，就是使用若干原有大小的 batch 数据进行训练，即通过“前向+反向”
网络计算得到梯度。其间会有一部分显存/内存用于存放梯度，然后对每个 batch 计算出的梯度进行叠加，当累加的次数达到某个预设值后，使用累加的梯度对模型进行参数更新，从而达到使用大 batch 数据训练的效果。

在较大的粒度上看， GM 是将训练一个 step 的过程由原来的 “前向 + 反向 + 更新” 改变成 “（前向 + 反向 + 梯度累加）x k + 更新”， 通过在最终更新前进行 k 次梯度的累加模拟出 batch size 扩大 k 倍的效果。
更具体细节可以参考 `《MG-WFBP: Efficient Data Communication for Distributed Synchronous SGD Algorithms》 <https://arxiv.org/abs/1811.11141>`__  。

三、动态图使用方法
--------------------------------

需要说明的是，动态图是天然支持 Gradient Merge。即，只要不调用 ``clear_gradient`` 方法，动态图的梯度会一直累积。
动态图下使用 Gradient Merge 的代码片段如下：

.. code-block::

   for batch_id, data in enumerate(train_loader()):
       ... ...
       avg_loss.backward()
       if batch_id % k == 0:
           optimizer.minimize(avg_loss)
           model.clear_gradients()
