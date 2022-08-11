
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

.. code-block:: python

    # Example1: set Regularizer in optimizer
    import paddle
    from paddle.regularizer import L1Decay
    import numpy as np
    linear = paddle.nn.Linear(10, 10)
    inp = paddle.rand(shape=[10, 10], dtype="float32")
    out = linear(inp)
    loss = paddle.mean(out)
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")
    momentum = paddle.optimizer.Momentum(
        learning_rate=0.1,
        parameters=linear.parameters(),
        weight_decay=L1Decay(0.0001))
    back = out.backward()
    momentum.step()
    momentum.clear_grad()


代码示例 2
::::::::::::

.. code-block:: python

    # Example2: set Regularizer in parameters
    # Set L1 regularization in parameters.
    # Global regularizer does not take effect on my_conv2d for this case.
    from paddle.nn import Conv2D
    from paddle import ParamAttr
    from paddle.regularizer import L2Decay

    my_conv2d = Conv2D(
                    in_channels=10,
                    out_channels=10,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(regularizer=L2Decay(coeff=0.01)),
                    bias_attr=False)
