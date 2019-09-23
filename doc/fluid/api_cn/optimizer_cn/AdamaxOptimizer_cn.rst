.. _cn_api_fluid_optimizer_AdamaxOptimizer:

AdamaxOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdamaxOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, regularization=None, name=None)

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
        adam = fluid.optimizer.Adamax(learning_rate=0.2)
        adam.minimize(loss)
     
    # Run the startup program once and only once.
    exe.run(startup_program)
     
    x = numpy.random.random(size=(10, 1)).astype('float32')
    outs = exe.run(program=train_program,
                  feed={'X': x},
                   fetch_list=[loss.name])

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










