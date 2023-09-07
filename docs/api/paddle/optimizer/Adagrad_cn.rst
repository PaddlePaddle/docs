.. _cn_api_paddle_optimizer_Adagrad:

Adagrad
-------------------------------

.. py:class:: paddle.optimizer.Adagrad(learning_rate, epsilon=1e-06, parameters=None, weight_decay=None, grad_clip=None, name=None, initial_accumulator_value=0.0)


Adaptive Gradient 优化器（自适应梯度优化器，简称 Adagrad）可以针对不同参数样本数不平均的问题，自适应地为各个参数分配不同的学习率。

其参数更新的计算过程如下：

.. math::

    moment\_out &= moment + grad * grad\\param\_out
    &= param - \frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}


相关论文：`Adaptive Subgradient Methods for Online Learning and Stochastic Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ 。

原始论文的算法中没有引入上述公式中的 ``epsilon`` 属性，此处引入该属性用于维持数值稳定性，避免除 0 错误发生。

引入 epsilon 参数依据：`Per-parameter adaptive learning rate methods <http://cs231n.github.io/neural-networks-3/#ada>`_ 。

参数
::::::::::::

    - **learning_rate** (float|Tensor) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个值为浮点型的 Tensor。
    - **epsilon** (float，可选) - 维持数值稳定性的浮点型值，默认值为 1e-06。
    - **parameters** (list，可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为 None，这时所有的参数都将被优化。
    - **weight_decay** (float|WeightDecayRegularizer，可选) - 正则化方法。可以是 float 类型的 L2 正则化系数或者正则化策略：:ref:`cn_api_fluid_regularizer_L1Decay` 、
      :ref:`cn_api_fluid_regularizer_L2Decay`。如果一个参数已经在 :ref:`cn_api_paddle_ParamAttr` 中设置了正则化，这里的正则化设置将被忽略；
      如果没有在 :ref:`cn_api_paddle_ParamAttr` 中设置正则化，这里的设置才会生效。默认值为 None，表示没有正则化。
    - **grad_clip** (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略：:ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_paddle_nn_ClipGradByGlobalNorm>` 、 :ref:`paddle.nn.ClipGradByNorm <cn_api_paddle_nn_ClipGradByNorm>` 、 :ref:`paddle.nn.ClipGradByValue <cn_api_paddle_nn_ClipGradByValue>` 。
      默认值为 None，此时将不进行梯度裁剪。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **initial_accumulator_value** (float，可选) - moment 累加器的初始值，默认值为 0.0。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.Adagrad
