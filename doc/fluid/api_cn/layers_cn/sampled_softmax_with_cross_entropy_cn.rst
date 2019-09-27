.. _cn_api_fluid_layers_sampled_softmax_with_cross_entropy:

sampled_softmax_with_cross_entropy
----------------------------------------------

.. py:function:: paddle.fluid.layers.sampled_softmax_with_cross_entropy(logits, label, num_samples, num_true=1, remove_accidental_hits=True, use_customized_samples=False, customized_samples=None, customized_probabilities=None, seed=0)

假设该OP输入logits的shape为[N, K]，N表示batch size大小。该OP首先对logits进行softmax计算，得到每个样例下分类的概率值，假设为logits_s，shape为[N，K]，其中样例个数为N，类别个数为K。

该OP的输入label的shape为[N，1]。对于某一个样例i的概率值logits_s[i]，其中有1个为正样例的概率，其他的K-1个为负样本的概率。

对于样例logits_s[i]，该OP并不会对该样例的所有输出为负样例的K-1个概率值计算交叉熵，而是从K-1个概率值中随机选择出num_samples。

将label真实标签的对应的概率值， 以及根据随机选择出的非真实标签对应的概率值拼接成[N，1 + num_samples] 大小的矩阵。
对此矩阵进行计算交叉熵损失即为该OP的输出。

参数：
        - **logits** （Variable）- logits是一个shape为[N x K]的2-D Tensor。N是batch size，K是类别数目。数据类型支持float32。
        - **label** （Variable）- 表示样例的真实标签。label是一个shape为[N，1]的真实标签。数据类型支持int64。
        - **num_samples** （int）- OP计算过程中，会随机从非真实标签的概率输出中选择出num_samples个概率进行计算交叉熵损失。
        - **num_true** （int）- 训练实例的目标类别总数，默认值为1。
        - **remove_accidental_hits** （bool，可选）- 随机选出num_samples个概率期间，可能会选出同一个非真实标签的概率值，如果此值为True，表示允许选择出多个同样的非真实标签的概率值出来。默认值为True。
        - **use_customized_samples** （bool，可选）- 此开关为开发人员测试使用，默认值为False。
        - **customized_samples** （Variable, 可选）- 此开关为开发人员测试使用，默认值为None。
        - **customized_probabilities** （Variable，可选）- 此开关为开发人员测试使用，默认值为None。
        - **seed** （int，可选）- 用于生成随机数的随机种子，在采样过程中使用。默认值为0。

返回：返回类型为Variable(Tensor|LoDTensor)，是一个2-D Tensor/LoDTensor，shape为[N x 1]。


<font color="#FF0000">**注意：此OP内部会进行Softmax运算。**</font>

**代码示例：**

.. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        input = fluid.layers.data(name='data', shape=[25], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        out = fluid.layers.sampled_softmax_with_cross_entropy(
                logits=input, label=label, num_samples=2) 

