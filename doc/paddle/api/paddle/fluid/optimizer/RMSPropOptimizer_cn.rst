.. _cn_api_fluid_optimizer_RMSPropOptimizer:

RMSPropOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.RMSPropOptimizer(learning_rate, rho=0.95, epsilon=1e-06, momentum=0.0, centered=False, parameter_list=None, regularization=None, grad_clip=None, name=None)




该接口实现均方根传播（RMSProp）法，是一种未发表的,自适应学习率的方法。原演示幻灯片中提出了RMSProp：[http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf]中的第29张。等式如下所示：

.. math::
    r(w, t) & = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\
    w & = w - \frac{\eta} {\sqrt{r(w,t) + \epsilon}} \nabla Q_{i}(w)
    
第一个等式计算每个权重平方梯度的移动平均值，然后将梯度除以 :math:`sqrtv（w，t）` 。

.. math::
   r(w, t) & = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\
   v(w, t) & = \beta v(w, t-1) +\frac{\eta} {\sqrt{r(w,t) +\epsilon}} \nabla Q_{i}(w)\\
         w & = w - v(w, t)

如果居中为真：

.. math::
      r(w, t) & = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\
      g(w, t) & = \rho g(w, t-1) + (1 -\rho)\nabla Q_{i}(w)\\
      v(w, t) & = \beta v(w, t-1) + \frac{\eta} {\sqrt{r(w,t) - (g(w, t))^2 +\epsilon}} \nabla Q_{i}(w)\\
            w & = w - v(w, t)
      
其中， :math:`ρ` 是超参数，典型值为0.9,0.95等。 :math:`beta` 是动量术语。  :math:`epsilon` 是一个平滑项，用于避免除零，通常设置在1e-4到1e-8的范围内。:math:`eta` 是学习率`learning_rate`
      
参数：
    - **learning_rate** （float） - 全局学习率。
    - **parameter_list** (list, 可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为None，这时所有的参数都将被优化。
    - **rho** （float，可选） - rho是等式中的 :math:`rho` ，默认值0.95。
    - **epsilon** （float，可选） - 等式中的epsilon是平滑项，避免被零除，默认值1e-6。
    - **momentum** （float，可选） - 方程中的β是动量项，默认值0.0。
    - **centered** （bool，可选） - 如果为True，则通过梯度的估计方差,对梯度进行归一化；如果False，则由未centered的第二个moment归一化。将此设置为True有助于模型训练，但会消耗额外计算和内存资源。默认为False。
    - **regularization** (WeightDecayRegularizer，可选) - 正则化方法。支持两种正则化策略: :ref:`cn_api_fluid_regularizer_L1Decay` 、 
      :ref:`cn_api_fluid_regularizer_L2Decay` 。如果一个参数已经在 :ref:`cn_api_fluid_ParamAttr` 中设置了正则化，这里的正则化设置将被忽略；
      如果没有在 :ref:`cn_api_fluid_ParamAttr` 中设置正则化，这里的设置才会生效。默认值为None，表示没有正则化。
    - **grad_clip** (GradientClipBase, 可选) – 梯度裁剪的策略，支持三种裁剪策略： :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。
      默认值为None，此时将不进行梯度裁剪。
    - **name** (str, 可选) - 可选的名称前缀，一般无需设置，默认值为None。
    
抛出异常:
    - ``ValueError`` -如果 ``learning_rate`` ， ``rho`` ， ``epsilon`` ， ``momentum`` 为None。

**示例代码**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
     
    place = fluid.CPUPlace()
    main = fluid.Program()
    with fluid.program_guard(main):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
        
        rms_optimizer = fluid.optimizer.RMSProp(learning_rate=0.1)
        rms_optimizer.minimize(avg_cost)
     
        fetch_list = [avg_cost]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)



.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None)

为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameter_list中的Parameters，最小化网络损失值loss。

参数：
    - **loss** (Variable) – 需要最小化的损失值变量
    - **startup_program** (Program, 可选) – 用于初始化parameter_list中参数的 :ref:`cn_api_fluid_Program` , 默认值为None，此时将使用 :ref:`cn_api_fluid_default_startup_program` 
    - **parameter_list** (list, 可选) – 待更新的Parameter或者Parameter.name组成的列表， 默认值为None，此时将更新所有的Parameter
    - **no_grad_set** (set, 可选) – 不需要更新的Parameter或者Parameter.name组成的集合，默认值为None
        
返回: tuple(optimize_ops, params_grads)，其中optimize_ops为参数优化OP列表；param_grads为由(param, param_grad)组成的列表，其中param和param_grad分别为参数和参数的梯度。该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。

返回类型： tuple

**示例代码**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
     
    place = fluid.CPUPlace()
    main = fluid.Program()
    with fluid.program_guard(main):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
        
        rms_optimizer = fluid.optimizer.RMSProp(learning_rate=0.1)
        rms_optimizer.minimize(avg_cost)
     
        fetch_list = [avg_cost]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)



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
        optimizer = fluid.optimizer.RMSPropOptimizer(learning_rate=0.01,
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

