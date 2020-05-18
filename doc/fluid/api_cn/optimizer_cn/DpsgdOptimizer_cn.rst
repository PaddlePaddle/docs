.. _cn_api_fluid_optimizer_DpsgdOptimizer:

DpsgdOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.DpsgdOptimizer(learning_rate=0.001, clip=0.9, batch_size=0.999, sigma=1e-8)




Dpsgd优化器是参考CCS16论文 `《Deep Learning with Differential Privacy》 <https://arxiv.org/abs/1607.00133>`_ 相关内容实现的。

其参数更新的计算公式如下:

.. math::
    g\_clip_t = \frac{g_t}{\max{(1, \frac{||g_t||^2}{clip})}}\\
.. math::
    g\_noise_t = g\_clip_t + \frac{gaussian\_noise(\sigma)}{batch\_size}\\
.. math::
    param\_out=param−learning\_rate*g\_noise_t


参数：
  - **learning_rate** (float|Variable，可选) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个值为浮点型的Variable，默认值为0.001
  - **clip** (float， 可选) - 裁剪梯度的L2正则项值的阈值下界，若梯度L2正则项值小于clip，则取clip作为梯度L2正则项值，默认值为0.9
  - **batch_size** (float， 可选) - 每个batch训练的样本数，默认值为0.999
  - **sigma** (float， 可选) - 参数更新时，会在梯度后添加一个满足高斯分布的噪声。此为高斯噪声的方差，默认值为1e-08

.. note::
    目前 ``DpsgdOptimizer`` 不支持 Sparse Parameter Optimization（稀疏参数优化）。

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
      optimizer = fluid.optimizer.Dpsgd(learning_rate=0.01, clip=10.0, batch_size=16.0, sigma=1.0)
      optimizer.minimize(loss)

    # Run the startup program once and only once.
    exe.run(startup_program)

    x = numpy.random.random(size=(10, 1)).astype('float32')
    outs = exe.run(program=train_program,
                feed={'X': x},
                 fetch_list=[loss.name])

.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None)

为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameter_list中的Parameters，最小化网络损失值loss。

参数：
    - **loss** (Variable) – 需要最小化的损失值变量
    - **startup_program** (Program， 可选) – 用于初始化parameter_list中参数的 :ref:`cn_api_fluid_Program` ， 默认值为None，此时将使用 :ref:`cn_api_fluid_default_startup_program`
    - **parameter_list** (list， 可选) – 待更新的Parameter或者Parameter.name组成的列表， 默认值为None，此时将更新所有的Parameter
    - **no_grad_set** (set， 可选) – 不需要更新的Parameter或者Parameter.name组成集合，默认值为None
         
返回: tuple(optimize_ops, params_grads)，其中optimize_ops为参数优化OP列表；param_grads为由(param, param_grad)组成的列表，其中param和param_grad分别为参数和参数的梯度。该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。

**代码示例**：

.. code-block:: python

    import numpy
    import paddle.fluid as fluid
     
    data = fluid.layers.data(name='X', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = fluid.layers.mean(hidden)
    adam = fluid.optimizer.Dpsgd(learning_rate=0.2)
    adam.minimize(loss)

    place = fluid.CPUPlace() # fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
     
    x = numpy.random.random(size=(10, 1)).astype('float32')
    exe.run(fluid.default_startup_program())
    outs = exe.run(program=fluid.default_main_program(),
                   feed={'X': x},
                   fetch_list=[loss.name])








