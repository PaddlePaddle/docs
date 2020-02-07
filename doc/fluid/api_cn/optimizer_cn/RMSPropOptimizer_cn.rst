.. _cn_api_fluid_optimizer_RMSPropOptimizer:

RMSPropOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.RMSPropOptimizer(learning_rate, rho=0.95, epsilon=1e-06, momentum=0.0, centered=False, regularization=None, name=None)

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
      
其中， :math:`ρ` 是超参数，典型值为0.9,0.95等。 :math:`beta` 是动量术语。  :math:`epsilon` 是一个平滑项，用于避免除零，通常设置在1e-4到1e-8的范围内。
      
参数：
    - **learning_rate** （float） - 全局学习率。
    - **rho** （float，可选） - rho是等式中的 :math:`rho` ，默认值0.95。
    - **epsilon** （float，可选） - 等式中的epsilon是平滑项，避免被零除，默认值1e-6。
    - **momentum** （float，可选） - 方程中的β是动量项，默认值0.0。
    - **centered** （bool，可选） - 如果为True，则通过梯度的估计方差,对梯度进行归一化；如果False，则由未centered的第二个moment归一化。将此设置为True有助于模型训练，但会消耗额外计算和内存资源。默认为False。
    - **regularization**  - 正则器项，如 ``fluid.regularizer.L2DecayRegularizer`` 。
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



.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None)

为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameter_list中的Parameters，最小化网络损失值loss。

参数：
    - **loss** (Variable) – 需要最小化的损失值变量
    - **startup_program** (Program, 可选) – 用于初始化parameter_list中参数的 :ref:`cn_api_fluid_Program` , 默认值为None，此时将使用 :ref:`cn_api_fluid_default_startup_program` 
    - **parameter_list** (list, 可选) – 待更新的Parameter或者Parameter.name组成的列表， 默认值为None，此时将更新所有的Parameter
    - **no_grad_set** (set, 可选) – 不需要更新的Parameter或者Parameter.name组成的集合，默认值为None
    - **grad_clip** (GradClipBase, 可选) – 梯度裁剪的策略，静态图模式不需要使用本参数，当前本参数只支持在dygraph模式下的梯度裁剪，未来本参数可能会调整，默认值为None

返回： (optimize_ops, params_grads)，数据类型为(list, list)，其中optimize_ops是minimize接口为网络添加的OP列表，params_grads是一个由(param, grad)变量对组成的列表，param是Parameter，grad是该Parameter对应的梯度值

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





