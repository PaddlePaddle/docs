
.. _cn_api_fluid_regularizer_L1Decay:

L1Decay
-------------------------------

.. py:attribute::   paddle.fluid.regularizer.L1Decay(regularization_coeff=0.0)

L1Decay实现L1权重衰减正则化，用于模型训练，使得权重矩阵稀疏。

具体实现中，L1权重衰减正则化的计算公式如下：

.. math::
            \\L1WeightDecay=reg\_coeff∗sign(parameter)\\

参数：
  - **regularization_coeff** (float) – L1正则化系数，默认值为0.0。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    main_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(main_prog, startup_prog):
        data = fluid.layers.data(name='image', shape=[3, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        hidden = fluid.layers.fc(input=data, size=128, act='relu')
        prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = fluid.layers.mean(loss)
    optimizer = fluid.optimizer.Adagrad(
        learning_rate=1e-4,
        regularization=fluid.regularizer.L1Decay(
            regularization_coeff=0.1))
    optimizer.minimize(avg_loss)



