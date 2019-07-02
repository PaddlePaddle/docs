###################
 fluid.regularizer
###################

.. _cn_api_fluid_regularizer_L1Decay:

L1Decay
-------------------------------

.. py:attribute::   paddle.fluid.regularizer.L1Decay

``L1DecayRegularizer`` 的别名



.. _cn_api_fluid_regularizer_L1DecayRegularizer:

L1DecayRegularizer
-------------------------------

.. py:class:: paddle.fluid.regularizer.L1DecayRegularizer(regularization_coeff=0.0)

实现 L1 权重衰减正则化。

L1正则将会稀疏化权重矩阵。


.. math::
            \\L1WeightDecay=reg\_coeff∗sign(parameter)\\

参数:
  - **regularization_coeff** (float) – 正则化系数
  
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
        regularization=fluid.regularizer.L1DecayRegularizer(
            regularization_coeff=0.1))
    optimizer.minimize(avg_loss)
    
  
  





.. _cn_api_fluid_regularizer_L2Decay:

L2Decay
-------------------------------

.. py:attribute::   paddle.fluid.regularizer.L2Decay

``L2DecayRegularizer`` 的别名





.. _cn_api_fluid_regularizer_L2DecayRegularizer:

L2DecayRegularizer
-------------------------------

.. py:class:: paddle.fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0)

实现L2 权重衰减正则化。 

较小的 L2 的有助于防止对训练数据的过度拟合。

.. math::
            \\L2WeightDecay=reg\_coeff*parameter\\

参数:
  - **regularization_coeff** (float) – 正则化系数
  
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
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.1))
    optimizer.minimize(avg_loss)







