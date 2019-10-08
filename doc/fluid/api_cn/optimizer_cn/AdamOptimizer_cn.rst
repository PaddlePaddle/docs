.. _cn_api_fluid_optimizer_AdamOptimizer:

AdamOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, regularization=None, name=None, lazy_mode=False)

Adam优化器出自 `Adam论文 <https://arxiv.org/abs/1412.6980>`_ 的第二节，能够利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。

其参数更新的计算公式如下：

.. math::
    \\t = t + 1
.. math::
    moment\_1\_out=\beta_1∗moment\_1+(1−\beta_1)∗grad
.. math::
    moment\_2\_out=\beta_2∗moment\_2+(1−\beta_2)∗grad*grad
.. math::
    learning\_rate=\frac{learning\_rate}{1-\beta_1^t}
.. math::
    param\_out=param-learning\_rate*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon}\\

相关论文：`Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_ 

参数: 
    - **learning_rate** (float|Variable，可选) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个值为浮点型的Variable，默认值为0.001
    - **beta1** (float, 可选) - 一阶矩估计的指数衰减率，默认值为0.9
    - **beta2** (float, 可选) - 二阶矩估计的指数衰减率，默认值为0.999
    - **epsilon** (float, 可选) - 保持数值稳定性的短浮点类型值，默认值为1e-08
    - **regularization** (WeightDecayRegularizer, 可选) - 正则化函数，用于减少泛化误差。例如可以是 :ref:`cn_api_fluid_regularizer_L2DecayRegularizer` ，默认值为None
    - **name** (str, 可选)- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None
    - **lazy_mode** （bool, 可选） - 设为True时，仅更新当前具有梯度的元素。官方Adam算法有两个移动平均累加器（moving-average accumulators）。累加器在每一步都会更新。在密集模式和稀疏模式下，两条移动平均线的每个元素都会更新。如果参数非常大，那么更新可能很慢。 lazy mode仅更新当前具有梯度的元素，所以它会更快。但是这种模式与原始的算法有不同的描述，可能会导致不同的结果，默认为False


**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
     
    place = fluid.CPUPlace()
    main = fluid.Program()
    with fluid.program_guard(main):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
        adam_optimizer = fluid.optimizer.AdamOptimizer(0.01)
        adam_optimizer.minimize(avg_cost)

        fetch_list = [avg_cost]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)


.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None)


通过更新parameter_list来添加操作，进而使损失最小化。

该算子相当于backward()和apply_gradients()功能的合体。

参数：
    - **loss** (Variable) – 用于优化过程的损失值变量
    - **startup_program** (Program) – 用于初始化在parameter_list中参数的startup_program
    - **parameter_list** (list) – 待更新的Variables组成的列表
    - **no_grad_set** (set|None) – 应该被无视的Variables集合
    - **grad_clip** (GradClipBase|None) – 梯度裁剪的策略

返回： (optimize_ops, params_grads)，分别为附加的算子列表；一个由(param, grad) 变量对组成的列表，用于优化

返回类型：   tuple








