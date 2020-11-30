.. _cn_api_fluid_optimizer_LambOptimizer:

LambOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.LambOptimizer(learning_rate=0.001, lamb_weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-06, parameter_list=None, regularization=None, grad_clip=None, exclude_from_weight_decay_fn=None, name=None)




LAMB（Layer-wise Adaptive Moments optimizer for Batching training）优化器
LAMB的优化器旨在不降低精度的前提下增大训练的批量大小，其支持自适应的逐元素更新和精确的分层校正。 更多信息请参考 `Large Batch Optimization for
Deep Learning: Training BERT in 76 minutes <https://arxiv.org/pdf/1904.00962.pdf>`_ 。
参数更新如下：

.. math::

    \begin{align}
    \begin{aligned}
     m_t &= \beta_1 m_{t - 1}+ (1 - \beta_1)g_t \\
     v_t &= \beta_2 v_{t - 1}  + (1 - \beta_2)g_t^2 \\
     r_t &= \frac{m_t}{\sqrt{v_t}+\epsilon} \\
     w_t &= w_{t-1} -\eta_t \frac{\left \| w_{t-1}\right \|}{\left \| r_t + \lambda w_{t-1}\right \|} (r_t + \lambda w_{t-1})
    \end{aligned}
    \end{align}

其中 :math:`m` 为第一个动量，:math:`v` 为第二个动量，:math:`\eta` 为学习率，:math:`\lambda` 为 LAMB 权重衰减率。

参数：
    - **learning_rate** (float|Variable) – 用于更新参数的学习率。可以是浮点数，或数据类型为浮点数的 Variable。
    - **lamb_weight_decay** (float) – LAMB权重衰减率。
    - **beta1** (float) – 第一个动量估计的指数衰减率。
    - **beta2** (float) – 第二个动量估计的指数衰减率。
    - **epsilon** (float) – 一个小的浮点值，目的是维持数值稳定性。
    - **parameter_list** (list, 可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为None，这时所有的参数都将被优化。
    - **regularization** (WeightDecayRegularizer，可选) - 正则化方法。支持两种正则化策略: :ref:`cn_api_fluid_regularizer_L1Decay` 、 
      :ref:`cn_api_fluid_regularizer_L2Decay` 。如果一个参数已经在 :ref:`cn_api_fluid_ParamAttr` 中设置了正则化，这里的正则化设置将被忽略；
      如果没有在 :ref:`cn_api_fluid_ParamAttr` 中设置正则化，这里的设置才会生效。默认值为None，表示没有正则化。
    - **grad_clip** (GradientClipBase, 可选) – 梯度裁剪的策略，支持三种裁剪策略： :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。
      默认值为None，此时将不进行梯度裁剪。
    - **exclude_from_weight_decay_fn** (function) – 当某个参数作为输入该函数返回值为 ``True`` 时，为该参数跳过权重衰减。 
    - **name** (str，可选) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值为None。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
     
    data = fluid.layers.data(name='x', shape=[5], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    cost = fluid.layers.mean(hidden)

    def exclude_fn(param):
        return param.name.endswith('.b_0')
     
    optimizer = fluid.optimizer.Lamb(learning_rate=0.002,
                                     exclude_from_weight_decay_fn=exclude_fn)
    optimizer.minimize(cost)


.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None)

为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameter_list中的Parameters，最小化网络损失值loss。

参数：
    - **loss** (Variable) – 需要最小化的损失值变量。
    - **startup_program** (Program, 可选) – 用于初始化parameter_list中参数的 :ref:`cn_api_fluid_Program` , 默认值为None，此时将使用 :ref:`cn_api_fluid_default_startup_program` 
    - **parameter_list** (list, 可选) – 待更新的Parameter或者Parameter.name组成的列表， 默认值为None，此时将更新所有的Parameter
    - **no_grad_set** (set, 可选) – 不需要更新的Parameter或者Parameter.name组成的的集合，默认值为None
         
返回: tuple(optimize_ops, params_grads)，其中optimize_ops为参数优化OP列表；param_grads为由(param, param_grad)组成的列表，其中param和param_grad分别为参数和参数的梯度。该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。

返回类型： tuple

**代码示例**：

.. code-block:: python

    import numpy
    import paddle.fluid as fluid
     
    x = fluid.layers.data(name='X', shape=[13], dtype='float32')
    y = fluid.layers.data(name='Y', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    loss = fluid.layers.mean(cost)
    adam = fluid.optimizer.LambOptimizer(learning_rate=0.2)
    adam.minimize(loss)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
     
    x = numpy.random.random(size=(10, 13)).astype('float32')
    y = numpy.random.random(size=(10, 1)).astype('float32')
    exe.run(fluid.default_startup_program())
    outs = exe.run(program=fluid.default_main_program(),
                   feed={'X': x, 'Y': y},
                   fetch_list=[loss.name])



.. py:method:: clear_gradients()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**


清除需要优化的参数的梯度。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    def exclude_fn(param):
        return param.name.endswith('.b_0')

    with fluid.dygraph.guard():
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = fluid.dygraph.to_variable(value)
        linear = fluid.Linear(13, 5, dtype="float32")
        optimizer = fluid.optimizer.LambOptimizer(learning_rate=0.02,
                                      exclude_from_weight_decay_fn=exclude_fn,
                                      parameter_list=linear.parameters())
        out = linear(a)
        out.backward()
        optimizer.minimize(out)
        optimizer.clear_gradients()

.. py:method:: set_lr()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**  

手动设置当前 ``optimizer`` 的学习率。当使用LearningRateDecay时，无法使用该API手动设置学习率，因为这将导致冲突。

参数：
    value (float|Variable) - 需要设置的学习率的值。

返回：无

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
            
    with fluid.dygraph.guard():
        linear = fluid.dygraph.nn.Linear(10, 10)
        adam = fluid.optimizer.Adam(0.1, parameter_list=linear.parameters())
        # 通过Python float数值手动设置学习率
        lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
        for i in range(5):
            adam.set_lr(lr_list[i])
            print("current lr is {}".format(adam.current_step_lr()))
        # 打印结果:
        #    current lr is 0.2
        #    current lr is 0.3
        #    current lr is 0.4
        #    current lr is 0.5
        #    current lr is 0.6


        # 通过 框架的Variable 设置学习率
        lr_var = fluid.layers.create_global_var(shape=[1], value=0.7, dtype='float32')
        adam.set_lr(lr_var)
        print("current lr is {}".format(adam.current_step_lr()))
        # 打印结果:
        #    current lr is 0.7



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

