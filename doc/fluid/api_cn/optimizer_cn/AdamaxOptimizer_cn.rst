.. _cn_api_fluid_optimizer_AdamaxOptimizer:

AdamaxOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdamaxOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, parameter_list=None, regularization=None, name=None)

Adamax优化器是参考 `Adam论文 <https://arxiv.org/abs/1412.6980>`_ 第7节Adamax优化相关内容所实现的。Adamax算法是基于无穷大范数的 `Adam <https://arxiv.org/abs/1412.6980>`_ 算法的一个变种，使学习率更新的算法更加稳定和简单。

其参数更新的计算公式如下:

.. math::
    \\t = t + 1
.. math::
    moment\_out=\beta_1∗moment+(1−\beta_1)∗grad
.. math::
    inf\_norm\_out=\max{(\beta_2∗inf\_norm+\epsilon, \left|grad\right|)}
.. math::
    learning\_rate=\frac{learning\_rate}{1-\beta_1^t}
.. math::
    param\_out=param−learning\_rate*\frac{moment\_out}{inf\_norm\_out}\\

相关论文：`Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_

论文中没有 ``epsilon`` 参数。但是，为了保持数值稳定性， 避免除0错误， 此处增加了这个参数。

参数：
  - **learning_rate** (float|Variable，可选) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个值为浮点型的Variable，默认值为0.001
  - **beta1** (float, 可选) - 一阶矩估计的指数衰减率，默认值为0.9
  - **beta2** (float, 可选) - 二阶矩估计的指数衰减率，默认值为0.999
  - **epsilon** (float, 可选) - 保持数值稳定性的短浮点类型值，默认值为1e-08
  - **parameter_list** (list, 可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为None，这时所有的参数都将被优化。
  - **regularization** (WeightDecayRegularizer, 可选) - 正则化函数，用于减少泛化误差。例如可以是 :ref:`cn_api_fluid_regularizer_L2DecayRegularizer` ，默认值为None
  - **name** (str, 可选)- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None

.. note::
    目前 ``AdamaxOptimizer`` 不支持 Sparse Parameter Optimization（稀疏参数优化）。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy
     
    # First create the Executor.
    place = fluid.CPUPlace() # fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
     
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        adam = fluid.optimizer.AdamaxOptimizer(learning_rate=0.2)
        adam.minimize(loss)
     
    # Run the startup program once and only once.
    exe.run(startup_program)
     
    x = numpy.random.random(size=(10, 1)).astype('float32')
    outs = exe.run(program=train_program,
                  feed={'X': x},
                   fetch_list=[loss.name])

.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None)

为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameter_list中的Parameters，最小化网络损失值loss。

参数：
    - **loss** (Variable) – 需要最小化的损失值变量
    - **startup_program** (Program, 可选) – 用于初始化parameter_list中参数的 :ref:`cn_api_fluid_Program` , 默认值为None，此时将使用 :ref:`cn_api_fluid_default_startup_program` 
    - **parameter_list** (list, 可选) – 待更新的Parameter或者Parameter.name组成的列表， 默认值为None，此时将更新所有的Parameter
    - **no_grad_set** (set, 可选) – 不需要更新的Parameter或者Parameter.name组成集合，默认值为None
    - **grad_clip** (GradClipBase, 可选) – 梯度裁剪的策略，静态图模式不需要使用本参数，当前本参数只支持在dygraph模式下的梯度裁剪，未来本参数可能会调整，默认值为None

返回： (optimize_ops, params_grads)，数据类型为(list, list)，其中optimize_ops是minimize接口为网络添加的OP列表，params_grads是一个由(param, grad)变量对组成的列表，param是Parameter，grad是该Parameter对应的梯度值

**代码示例**：

.. code-block:: python

    import numpy
    import paddle.fluid as fluid
     
    data = fluid.layers.data(name='X', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = fluid.layers.mean(hidden)
    adam = fluid.optimizer.Adamax(learning_rate=0.2)
    adam.minimize(loss)

    place = fluid.CPUPlace() # fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
     
    x = numpy.random.random(size=(10, 1)).astype('float32')
    exe.run(fluid.default_startup_program())
    outs = exe.run(program=fluid.default_main_program(),
                   feed={'X': x},
                   fetch_list=[loss.name])



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
        optimizer = fluid.optimizer.AdamaxOptimizer(learning_rate=0.2,
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

