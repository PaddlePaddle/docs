.. _cn_api_paddle_regularizer_L2Decay:

L2Decay
-------------------------------

.. py:attribute::   paddle.regularizer.L2Decay(coeff=0.0)


L2Decay 实现 L2 权重衰减正则化，用于模型训练，有助于防止模型对训练数据过拟合。

该类生成的实例对象，需要设置在 :ref:`cn_api_paddle_ParamAttr` 或者 ``optimizer``
(例如 :ref:`cn_api_paddle_optimizer_Momentum` )中，在 ``ParamAttr`` 中设置时，
只对该网络层中的参数生效；在 ``optimizer`` 中设置时，会对所有的参数生效；如果同时设置，
在 ``ParamAttr`` 中设置的优先级会高于在 ``optimizer`` 中设置，即，对于一个可训练的参数，如果在
``ParamAttr`` 中定义了正则化，那么会忽略 ``optimizer`` 中的正则化；否则会使用 ``optimizer`` 中的
正则化。

具体实现中，L2 权重衰减正则化的损失函数计算如下：

.. math::
            \\loss = 0.5 * coeff * reduce\_sum(square(x))\\

参数
::::::::::::

  - **coeff** (float) – 正则化系数，默认值为 0.0。

代码示例 1
::::::::::::

COPY-FROM: paddle.regularizer.L2Decay:code-example1


代码示例 2
::::::::::::

COPY-FROM: paddle.regularizer.L2Decay:code-example2
