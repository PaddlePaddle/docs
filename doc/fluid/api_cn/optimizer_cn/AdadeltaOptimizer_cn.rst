.. _cn_api_fluid_optimizer_AdadeltaOptimizer:

AdadeltaOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdadeltaOptimizer(learning_rate, epsilon=1.0e-6, rho=0.95, regularization=None, name=None)

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
    - **regularization** (WeightDecayRegularizer，可选) - 正则化方法，例如fluid.regularizer.L2DecayRegularizer等。默认值为None，表示无正则化。
    - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

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
    - **parameter_list** (list(Variable)，可选) – 待更新的参数列表。默认值为None，表示所有参数均需要更新。
    - **no_grad_set** (set，可选) – 无需计算梯度的变量集合。默认值为None，表示所有变量均需计算梯度。
    - **grad_clip** (GradClipBase，可选) – 梯度裁剪的策略，目前仅在动态图模式下有效。

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


