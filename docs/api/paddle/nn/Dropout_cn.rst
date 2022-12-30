.. _cn_api_nn_Dropout:

Dropout
-------------------------------

.. py:function:: paddle.nn.Dropout(p=0.5, axis=None, mode="upscale_in_train", name=None)

Dropout 是一种正则化手段，根据给定的丢弃概率 `p`，在训练过程中随机将一些神经元输出设置为 0，通过阻止神经元节点间的相关性来减少过拟合。论文请参考：`Improving neural networks by preventing co-adaptation of feature detectors <https://arxiv.org/abs/1207.0580>`_

在动态图模式下，请使用模型的 `eval()` 方法切换至测试阶段。

.. note::
   对应的 `functional 方法` 请参考：:ref:`cn_api_nn_functional_dropout` 。

参数
:::::::::
 - **p** (float，可选) - 将输入节点置为 0 的概率，即丢弃概率。默认值为 0.5。
 - **axis** (int|list，可选) - 指定对输入 `Tensor` 进行 Dropout 操作的轴。默认值为 None。
 - **mode** (str，可选) - 丢弃单元的方式，有 'upscale_in_train' 和 'downscale_in_infer' 两种可供选择，默认值为 'upscale_in_train'。计算方法如下：

    1. upscale_in_train（默认值），在训练时增大输出结果：

       - 训练时： :math:`out = input \times \frac{mask}{(1.0 - p)}`
       - 预测时： :math:`out = input`

    2. downscale_in_infer，在预测时减小输出结果：

       - 训练时： :math:`out = input \times mask`
       - 预测时： :math:`out = input \times (1.0 - p)`

 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
 - **输入** : N-D `Tensor` 。
 - **输出** : N-D `Tensor`，形状与输入相同。

代码示例
:::::::::

COPY-FROM: paddle.nn.Dropout
