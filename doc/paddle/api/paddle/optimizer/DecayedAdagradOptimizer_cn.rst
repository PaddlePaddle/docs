.. _cn_api_fluid_optimizer_DecayedAdagradOptimizer:

DecayedAdagradOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.DecayedAdagradOptimizer(learning_rate, decay=0.95, epsilon=1e-06, parameter_list=None, regularization=None, grad_clip=None, name=None)




Decayed Adagrad优化器，可以看做是引入了衰减率的 `Adagrad <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ 算法，用于解决使用 :ref:`cn_api_fluid_optimizer_AdagradOptimizer` 优化器时，在模型训练中后期学习率急剧下降的问题。

其参数更新的计算公式如下：

.. math::
    moment\_out = decay*moment+(1-decay)*grad*grad
.. math::
    param\_out = param-\frac{learning\_rate*grad}{\sqrt{moment\_out}+\epsilon }

在原论文中没有 ``epsilon`` 参数。但是，为了保持数值稳定性， 防止除0错误， 此处增加了这个参数。

相关论文：`Adaptive Subgradient Methods for Online Learning and Stochastic Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_

    
参数：
  - **learning_rate** (float|Variable) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个值为浮点型的Variable
  - **parameter_list** (list, 可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为None，这时所有的参数都将被优化。
  - **decay** (float，可选) – 衰减率，默认值为0.95
  - **regularization** (WeightDecayRegularizer，可选) - 正则化方法。支持两种正则化策略: :ref:`cn_api_fluid_regularizer_L1Decay` 、 
    :ref:`cn_api_fluid_regularizer_L2Decay` 。如果一个参数已经在 :ref:`cn_api_fluid_ParamAttr` 中设置了正则化，这里的正则化设置将被忽略；
    如果没有在 :ref:`cn_api_fluid_ParamAttr` 中设置正则化，这里的设置才会生效。默认值为None，表示没有正则化。
  - **grad_clip** (GradientClipBase, 可选) – 梯度裁剪的策略，支持三种裁剪策略： :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。
    默认值为None，此时将不进行梯度裁剪。
  - **epsilon** (float，可选) - 保持数值稳定性的短浮点类型值，默认值为1e-06
  - **name** (str, 可选)- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None

.. note::
    当前， ``DecayedAdagradOptimizer`` 不支持Sparse Parameter Optimization（稀疏参数优化）
  
**代码示例**
 
.. code-block:: python
        
    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    from paddle.fluid.optimizer import DecayedAdagrad
        
    x = layers.data( name='x', shape=[-1, 10], dtype='float32' )
    trans = layers.fc( x, 100 )
    cost = layers.reduce_mean( trans )
    optimizer = fluid.optimizer.DecayedAdagradOptimizer(learning_rate=0.2)
    optimizer.minimize(cost)

.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None)

为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameter_list中的Parameters，最小化网络损失值loss。

参数：
    - **loss** (Variable) – 需要最小化的损失值变量
    - **startup_program** (Program, 可选) – 用于初始化parameter_list中参数的 :ref:`cn_api_fluid_Program` , 默认值为None，此时将使用 :ref:`cn_api_fluid_default_startup_program`
    - **parameter_list** (list, 可选) – 待更新的Parameter或者Parameter.name组成的列表， 默认值为None，此时将更新所有的Parameter
    - **no_grad_set** (set, 可选) – 不需要更新的Parameter或者Parameter.name组成的集合，默认值为None
        
返回: tuple(optimize_ops, params_grads)，其中optimize_ops为参数优化OP列表；param_grads为由(param, param_grad)组成的列表，其中param和param_grad分别为参数和参数的梯度。该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。

返回类型： tuple

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid
     
    inp = fluid.layers.data(
        name="inp", shape=[2, 2], append_batch_size=False)
    out = fluid.layers.fc(inp, size=3)
    out = fluid.layers.reduce_sum(out)
    optimizer = fluid.optimizer.DecayedAdagrad(learning_rate=0.2)
    optimizer.minimize(out)

    np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    exe.run(
        feed={"inp": np_inp},
        fetch_list=[out.name])


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
        optimizer = fluid.optimizer.DecayedAdagradOptimizer(learning_rate=0.02,
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

