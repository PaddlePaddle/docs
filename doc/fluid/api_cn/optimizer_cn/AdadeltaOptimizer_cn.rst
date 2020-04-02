.. _cn_api_fluid_optimizer_AdadeltaOptimizer:

AdadeltaOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdadeltaOptimizer(learning_rate, epsilon=1.0e-6, rho=0.95, parameter_list=None, regularization=None, name=None)

**注意：此接口不支持稀疏参数更新。**

Adadelta优化器，具体细节可参考论文 `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/abs/1212.5701>`_ 。

更新公式如下：

.. math::

    E(g_t^2) &= \rho * E(g_{t-1}^2) + (1-\rho) * g^2\\
    learning\_rate &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }\\
    E(dx_t^2) &= \rho * E(dx_{t-1}^2) + (1-\rho) * (-g*learning\_rate)^2


参数：
    - **learning_rate** (float|Variable) - 全局学习率。
    - **epsilon** (float) - 维持数值稳定性的浮点型值，默认值为1.0e-6。
    - **rho** (float) - 算法中的衰减率，默认值为0.95。
    - **parameter_list** (list, 可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为None，这时所有的参数都将被优化。
    - **regularization** (WeightDecayRegularizer，可选) - 正则化方法，例如fluid.regularizer.L2DecayRegularizer等。默认值为None，表示无正则化。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    image = fluid.layers.data(name='image', shape=[28], dtype='float32')
    fc = fluid.layers.fc(image, size=10)
    cost = fluid.layers.reduce_mean(fc)
    optimizer = fluid.optimizer.AdadeltaOptimizer(
        learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)
    optimizer_ops, params_grads = optimizer.minimize(cost)


.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None)

为训练网络添加反向和参数优化部分，进而使损失最小化。

参数：
    - **loss** (Variable) – 优化器的损失变量。
    - **startup_program** (Program，可选) – 参数所在的startup program。默认值为None，表示 :ref:`cn_api_fluid_default_startup_program` 。
    - **parameter_list** (list，可选) – 待更新的Parameter或者Parameter.name组成的列表。默认值为None，表示所有参数均需要更新。
    - **no_grad_set** (set，可选) – 不需要更新的Parameter或者Parameter.name组成的集合。默认值为None。
    - **grad_clip** (GradientClipBase, 可选) – 梯度裁剪的策略，支持三种裁剪策略： :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。
      默认值为None，此时将不进行梯度裁剪。

返回: tuple(optimize_ops, params_grads)，其中optimize_ops为参数优化OP列表；param_grads为由(param, param_grad)组成的列表，其中param和param_grad分别为参数和参数的梯度。

返回类型: tuple

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    image = fluid.layers.data(name='image', shape=[28], dtype='float32')
    fc = fluid.layers.fc(image, size=10)
    cost = fluid.layers.reduce_mean(fc)
    optimizer = fluid.optimizer.AdadeltaOptimizer(
        learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)
    optimizer_ops, params_grads = optimizer.minimize(cost)


.. py:method:: clear_gradients()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

清除需要优化的参数的梯度。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    with fluid.dygraph.guard():
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = fluid.dygraph.to_variable(value)
        linear = fluid.Linear(13, 5, dtype="float32")
        optimizer = fluid.optimizer.AdadeltaOptimizer(learning_rate=0.0003, epsilon=1.0e-6, rho=0.95,
                                                      parameter_list=linear.parameters())
        out = linear(a)
        out.backward()
        optimizer.minimize(out)
        optimizer.clear_gradients()


.. py:method:: current_step_lr()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

获取当前步骤的学习率。当不使用LearningRateDecay时，每次调用的返回值都相同，否则返回当前步骤的学习率。

返回：当前步骤的学习率。

返回类型：float

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    # example1: LearningRateDecay is not used, return value is all the same
    with fluid.dygraph.guard():
        emb = fluid.dygraph.Embedding([10, 10])
        adam = fluid.optimizer.Adam(0.001, parameter_list = emb.parameters())
        lr = adam.current_step_lr()
        print(lr) # 0.001

    # example2: PiecewiseDecay is used, return the step learning rate
    with fluid.dygraph.guard():
        inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
        linear = fluid.dygraph.nn.Linear(10, 10)
        inp = fluid.dygraph.to_variable(inp)
        out = linear(inp)
        loss = fluid.layers.reduce_mean(out)

        bd = [2, 4, 6, 8]
        value = [0.2, 0.4, 0.6, 0.8, 1.0]
        adam = fluid.optimizer.Adam(fluid.dygraph.PiecewiseDecay(bd, value, 0),
                           parameter_list=linear.parameters())

        # first step: learning rate is 0.2
        np.allclose(adam.current_step_lr(), 0.2, rtol=1e-06, atol=0.0) # True

        # learning rate for different steps
        ret = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
        for i in range(12):
            adam.minimize(loss)
            lr = adam.current_step_lr()
            np.allclose(lr, ret[i], rtol=1e-06, atol=0.0) # True

