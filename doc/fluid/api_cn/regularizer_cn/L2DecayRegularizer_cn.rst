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







