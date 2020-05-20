.. _cn_api_fluid_regularizer_L2Decay:

L2Decay
-------------------------------

.. py:attribute::   paddle.fluid.regularizer.L2Decay




L2Decay实现L2权重衰减正则化，用于模型训练，有助于防止模型对训练数据过拟合。

该类生成的实例对象，需要设置在 :ref:`cn_api_fluid_ParamAttr` 或者 ``optimizer`` 
(例如 :ref:`cn_api_fluid_optimizer_SGDOptimizer` )中，在 ``ParamAttr`` 中设置时，
只对该网络层中的参数生效；在 ``optimizer`` 中设置时，会对所有的参数生效；如果同时设置，
在 ``ParamAttr`` 中设置的优先级会高于在 ``optimizer`` 中设置。

具体实现中，L2权重衰减正则化的计算公式如下：

.. math::
            \\L2WeightDecay=reg\_coeff*parameter\\

参数:
  - **regularization_coeff** (float) – 正则化系数，默认值为0.0。

**代码示例1**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_prog = paddle.Program()
    startup_prog = paddle.Program()
    with paddle.program_guard(main_prog, startup_prog):
        data = fluid.layers.data(name='image', shape=[3, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        hidden = fluid.layers.fc(input=data, size=128, act='relu')
        prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = paddle.mean(loss)
    optimizer = paddle.optimizer.Adagrad(learning_rate=0.0001, regularization=
        fluid.regularizer.L2Decay(regularization_coeff=0.1))
    optimizer.minimize(avg_loss)

**代码示例2**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_prog = paddle.Program()
    startup_prog = paddle.Program()
    with paddle.program_guard(main_prog, startup_prog):
        data = fluid.layers.data(name='image', shape=[3, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        hidden = fluid.layers.fc(input=data, size=128, act='relu')
        prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = paddle.mean(loss)
    optimizer = paddle.optimizer.Adagrad(learning_rate=0.0001, regularization=
        fluid.regularizer.L2Decay(regularization_coeff=0.1))
    optimizer.minimize(avg_loss)

