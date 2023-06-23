
.. _cn_api_paddle_regularizer_L1Decay:

L1Decay
-------------------------------

.. py:attribute::   paddle.regularizer.L1Decay(coeff=0.0)

L1Decay 实现 L1 权重衰减正则化，用于模型训练，使得权重矩阵稀疏。

该类生成的实例对象，需要设置在 :ref:`cn_api_paddle_ParamAttr` 或者 ``optimizer``
(例如 :ref:`cn_api_paddle_optimizer_Momentum` )中，在 ``ParamAttr`` 中设置时，只对该
网络层中的可训练参数生效；在 ``optimizer`` 中设置时，会对所有的可训练参数生效；如果同时设置，在
``ParamAttr`` 中设置的优先级会高于在 ``optimizer`` 中的设置，即，对于一个可训练的参数，如果在
``ParamAttr`` 中定义了正则化，那么会忽略 ``optimizer`` 中的正则化；否则会使用 ``optimizer``中的
正则化。

具体实现中，L1 权重衰减正则化的损失函数计算如下：

.. math::
            \\loss = coeff * reduce\_sum(abs(x))\\

参数
::::::::::::

  - **coeff** (float) – L1 正则化系数，默认值为 0.0。

代码示例 1
::::::::::::

COPY-FROM: paddle.regularizer.L1Decay:code-example1


代码示例 2
::::::::::::

COPY-FROM: paddle.regularizer.L1Decay:code-example2
