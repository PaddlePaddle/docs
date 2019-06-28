#################
 fluid.optimizer
#################

.. _cn_api_fluid_optimizer_Adadelta:

Adadelta
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.Adadelta

``AdadeltaOptimizer`` 的别名






.. _cn_api_fluid_optimizer_Adagrad:

Adagrad
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.Adagrad

``AdagradOptimizer`` 的别名




.. _cn_api_fluid_optimizer_AdagradOptimizer:

AdagradOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdagradOptimizer(learning_rate, epsilon=1e-06, regularization=None, name=None, initial_accumulator_value=0.0)

**Adaptive Gradient Algorithm(Adagrad)**

更新如下：

.. math::

    moment\_out &= moment + grad * grad\\param\_out 
    &= param - \frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}

原始论文（http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf）没有epsilon属性。在我们的实现中也作了如下更新：
http://cs231n.github.io/neural-networks-3/#ada 用于维持数值稳定性，避免除数为0的错误发生。

参数：
    - **learning_rate** (float|Variable)-学习率，用于更新参数。作为数据参数，可以是一个浮点类型值或者有一个浮点类型值的变量
    - **epsilon** (float) - 维持数值稳定性的短浮点型值
    - **regularization** - 规则化函数，例如fluid.regularizer.L2DecayRegularizer
    - **name** - 名称前缀（可选）
    - **initial_accumulator_value** (float) - moment累加器的初始值。

**代码示例**：

.. code-block:: python:

    import paddle.fluid as fluid
    import numpy as np
     
    np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    inp = fluid.layers.data(
        name="inp", shape=[2, 2], append_batch_size=False)
    out = fluid.layers.fc(inp, size=3)
    out = fluid.layers.reduce_sum(out)
    optimizer = fluid.optimizer.Adagrad(learning_rate=0.2)
    optimizer.minimize(out)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    exe.run(
        feed={"inp": np_inp},
        fetch_list=[out.name])






.. _cn_api_fluid_optimizer_Adam:

Adam
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.Adam

``AdamOptimizer`` 的别名





.. _cn_api_fluid_optimizer_Adamax:

Adamax
-------------------------------

.. py:attribute:: paddle.fluid.optimizer.Adamax

``AdamaxOptimizer`` 的别名






.. _cn_api_fluid_optimizer_AdamaxOptimizer:

AdamaxOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdamaxOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, regularization=None, name=None)

我们参考Adam论文第7节中的Adamax优化: https://arxiv.org/abs/1412.6980 ， Adamax是基于无穷大范数的Adam算法的一个变种。


Adamax 更新规则:

.. math::
    \\t = t + 1
.. math::
    moment\_out=\beta_1∗moment+(1−\beta_1)∗grad
.. math::
    inf\_norm\_out=\max{(\beta_2∗inf\_norm+ϵ, \left|grad\right|)}
.. math::
    learning\_rate=\frac{learning\_rate}{1-\beta_1^t}
.. math::
    param\_out=param−learning\_rate*\frac{moment\_out}{inf\_norm\_out}\\


论文中没有 ``epsilon`` 参数。但是，为了数值稳定性， 防止除0错误， 增加了这个参数

**代码示例**：

.. code-block:: python:

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

参数:
  - **learning_rate**  (float|Variable) - 用于更新参数的学习率。可以是浮点值，也可以是具有一个浮点值作为数据元素的变量。
  - **beta1** (float) - 第1阶段估计的指数衰减率
  - **beta2** (float) - 第2阶段估计的指数衰减率。
  - **epsilon** (float) -非常小的浮点值，为了数值的稳定性质
  - **regularization** - 正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** - 可选的名称前缀。

.. note::
    目前 ``AdamaxOptimizer`` 不支持  sparse parameter optimization.

  










.. _cn_api_fluid_optimizer_AdamOptimizer:

AdamOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, regularization=None, name=None, lazy_mode=False)

该函数实现了自适应矩估计优化器，介绍自 `Adam论文 <https://arxiv.org/abs/1412.6980>`_ 的第二节。Adam是一阶基于梯度下降的算法，基于自适应低阶矩估计。
Adam更新如下：

.. math::

    t & = t + 1\\moment\_out & = {\beta}_1 * moment + (1 - {\beta}_1) * grad\\inf\_norm\_out & = max({\beta}_2 * inf\_norm + \epsilon, |grad|)\\learning\_rate & = \frac{learning\_rate}{1 - {\beta}_1^t}\\param\_out & = param - learning\_rate * \frac{moment\_out}{inf\_norm\_out}

参数: 
    - **learning_rate** (float|Variable)-学习率，用于更新参数。作为数据参数，可以是一个浮点类型值或有一个浮点类型值的变量
    - **beta1** (float)-一阶矩估计的指数衰减率
    - **beta2** (float)-二阶矩估计的指数衰减率
    - **epsilon** (float)-保持数值稳定性的短浮点类型值
    - **regularization** - 规则化函数，例如''fluid.regularizer.L2DecayRegularizer
    - **name** - 可选名称前缀
    - **lazy_mode** （bool: false） - 官方Adam算法有两个移动平均累加器（moving-average accumulators）。累加器在每一步都会更新。在密集模式和稀疏模式下，两条移动平均线的每个元素都会更新。如果参数非常大，那么更新可能很慢。 lazy mode仅更新当前具有梯度的元素，所以它会更快。但是这种模式与原始的算法有不同的描述，可能会导致不同的结果。


**代码示例**：

.. code-block:: python:

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








.. _cn_api_fluid_optimizer_DecayedAdagrad:

DecayedAdagrad
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.DecayedAdagrad

``DecayedAdagradOptimizer`` 的别名





.. _cn_api_fluid_optimizer_DecayedAdagradOptimizer:

DecayedAdagradOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.DecayedAdagradOptimizer(learning_rate, decay=0.95, epsilon=1e-06, regularization=None, name=None)

Decayed Adagrad Optimizer

`原始论文 <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_

原始论文： `http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_  中没有 ``epsilon`` 参数。但是，为了数值稳定性， 防止除0错误， 增加了这个参数

.. math::
    moment\_out = decay*moment+(1-decay)*grad*grad
.. math::
    param\_out=param-\frac{learning\_rate*grad}{\sqrt{moment\_out+\epsilon }}
    
参数:
  - **learning_rate** (float|Variable) - 用于更新参数的学习率。可以是浮点值，也可以是具有一个浮点值作为数据元素的变量。
  - **decay** (float) – 衰减率
  - **regularization** - 一个正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **epsilon** (float) - 非常小的浮点值，为了数值稳定性
  - **name** — 可选的名称前缀。

  
**代码示例**
 
.. code-block:: python
        
  import paddle.fluid as fluid
  import paddle.fluid.layers as layers
  from paddle.fluid.optimizer import DecayedAdagrad
     
  x = layers.data( name='x', shape=[-1, 10], dtype='float32' )
  trans = layers.fc( x, 100 )
  cost = layers.reduce_mean( trans )
  optimizer = fluid.optimizer.DecayedAdagrad(learning_rate=0.2)
  optimizer.minimize(cost)

.. note::
  当前， ``DecayedAdagradOptimizer`` 不支持 sparse parameter optimization




.. _cn_api_fluid_optimizer_DGCMomentumOptimizer:

DGCMomentumOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.DGCMomentumOptimizer(learning_rate, momentum, rampup_begin_step, rampup_step=1, sparsity=[0.999], use_nesterov=False, local_grad_clip_norm=None, num_trainers=None, regularization=None, name=None)

原始论文: https://arxiv.org/abs/1712.01887

DGC通过仅发送重要梯度（稀疏更新）来减少通信带宽：仅发送大于给定阈值的梯度。

为避免丢失信息，DGC在本地累积其余梯度。最终，这些梯度会积累到足够大，从而可以传输。

因此，DGC即时发送相对较大的梯度，但最终随时间积累而发送所有梯度。

此外，为了确保不损失精度，DGC在梯度稀疏化之上采用动量修正和局部梯度修剪(clip)来维持模型性能。

DGC还使用动量因子掩藏(momentum factor masking)和预训练(warm-up)来克服由于reduced通讯而导致的数据陈旧性(staleness)问题。

这个优化器会执行如下操作：

1. 通过从张量获取前TopK个导入值来压缩梯度，并将其用于allreduce以减少网络带宽。
2. 调用momentum来降低cost。

参数: 
    - **learning_rate** （float | Variable） - 用于更新参数的学习率。可以是浮点值或由一个浮点型数据组成的Variable。
    - **momentum** （float） - 动量因子。
    - **rampup_begin_step** （int） - 进行梯度压缩的起步点。
    - **rampup_step** （int） - 使用稀疏期的时间。默认值为1.例如：如果稀疏度为[0.75,0.9375,0.984375,0.996,0.999]，并且rampup_step为5，则在0步时使用0.75，在1步时使用0.9375，依此类推。当达到sparsity数组末尾时，它此后延续使用0.999。
    - **sparsity** （list [float]） - 从梯度张量中获取较为重要的元素，比率为（1-当前稀疏度）。
    - **use_nesterov** （bool） - 启用Nesterov momentum。 True意味着使用nesterov。
    - **local_grad_clip_norm** （float） - 如果需要，clip norm值。
    - **num_trainers**   - 训练节点的数量。
    - **regularization**  - 正则器，如fluid.regularizer.L2DecayRegularizer。
    - **name**   - 可选的名称前缀。

**代码示例**

.. code-block:: python

    optimizer = fluid.optimizer.DGCMomentumOptimizer(
                                        learning_rate=0.0001,
                                        momentum=0.9,
                                        rampup_step=1000,
                                        rampup_begin_step=1252,
                                        sparsity=[0.999, 0.999])



.. _cn_api_fluid_optimizer_PipelineOptimizer:

PipelineOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.PipelineOptimizer(optimizer, cut_list=None, place_list=None, concurrency_list=None, queue_size=30, sync_steps=1, start_cpu_core_id=0)

Pipeline 优化器训练。该程序将由cut_list分割。如果cut_list的长度是k，则整个程序（包括向后部分）将被分割为2 * k-1个部分。 所以place_list和concurrency_list的长度也必须是2 * k-1。 

.. note::

    虽然异步模式应用于管道训练中以加速，但最终的性能取决于每个管道的训练进度。 我们将在未来尝试同步模式。

参数:
    - **optimizer** (Optimizer) - 基础优化器，如SGD
    - **cut_list** (list of Variable list) - main_program的cut变量
    - **place_lis** (list of Place) - 某部分运行的位置
    - **concurrency_lis** (list of int) - 并发度
    - **queue_size** (int) - 每个部分都将使用其范围内队列(in-scope queue)中的范围并将范围生成到范围外队列(out-scope queue)。 而这个参数定范围队列大小。 这一参数可选，默认值：30。
    - **sync_steps** (int) - 不同显卡之间的同步步数
    - **start_cpu_core_id** (int) - 设置第一个cpu核的id。这一参数可选，默认值：0。

**代码示例**

.. code-block:: python

        x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
        y = fluid.layers.data(name='y', shape=[1], dtype='int64', lod_level=0)
        emb_x = layers.embedding(input=x, param_attr=fluid.ParamAttr(name="embx"), size=[10,2], is_sparse=False)
        emb_y = layers.embedding(input=y, param_attr=fluid.ParamAttr(name="emby",learning_rate=0.9), size=[10,2], is_sparse=False)
        concat = layers.concat([emb_x, emb_y], axis=1)
        fc = layers.fc(input=concat, name="fc", size=1, num_flatten_dims=1, bias_attr=False)
        loss = layers.reduce_mean(fc)
        optimizer = fluid.optimizer.SGD(learning_rate=0.5)
        optimizer = fluid.optimizer.PipelineOptimizer(optimizer,
                cut_list=[[emb_x, emb_y], [loss]],
                place_list=[fluid.CPUPlace(), fluid.CUDAPlace(0), fluid.CPUPlace()],
                concurrency_list=[1, 1, 4],
                queue_size=2,
                sync_steps=1,
                )
        optimizer.minimize(loss)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        filelist = [] # 您应该根据需求自行设置文件列表, 如: filelist = ["dataA.txt"]
        dataset = fluid.DatasetFactory().create_dataset("FileInstantDataset")
        dataset.set_use_var([x,y])
        dataset.set_batch_size(batch_size)
        dataset.set_filelist(filelist)
        exe.train_from_dataset(
                    fluid.default_main_program(),
                    dataset,
                    thread=2,
                    debug=False,
                    fetch_list=[],
                    fetch_info=[],
                    print_period=1)


.. py:method:: extract_section_opt_ops(ops, cut_point_name)
    
获取指定section的优化算子(opt ops)

.. py:method:: extract_section_opt_ops(ops, cut_point_name)
  
获取指定section的输入和输出

.. py:method:: find_persistable_vars(ops, whole_parameters)

获取指定section的持久性输入变量

.. py:method:: extract_section_ops(ops, cut_point_name)

获取指定的section的算子(ops)



.. _cn_api_fluid_optimizer_ExponentialMovingAverage:

ExponentialMovingAverage
-------------------------------

.. py:class:: paddle.fluid.optimizer.ExponentialMovingAverage(decay=0.999, thres_steps=None, name=None)

用指数衰减计算参数的移动平均值。
给出参数 :math:`\theta` ，它的指数移动平均值(exponential moving average, EMA)
为

.. math::
    \begin{align}\begin{aligned}\text{EMA}_0 & = 0\\\text{EMA}_t & = \text{decay} * \text{EMA}_{t-1} + (1 - \text{decay}) * \theta_t\end{aligned}\end{align}


用 ``update()`` 方法计算出的平均结果将保存在由对象创建和维护的临时变量中，并且可以通过调用 ``apply()`` 方法把结果应用于当前模型的参数。另外，``restore()`` 方法用于恢复参数。

**偏差教正。**  所有的EMAs均初始化为 :math:`0` ，因此它们将为零偏差，可以通过除以因子 :math:`(1 - \text{decay}^t)` 来校正，即在调用 ``apply()`` 方法时应用于参数的真实EMAs将为：

.. math::
    \widehat{\text{EMA}}_t = \frac{\text{EMA}_t}{1 - \text{decay}^t}

**衰减率调度。**  一个非常接近于1的很大的衰减率将会导致平均值移动得很慢。更优的策略是，一开始就设置一个相对较小的衰减率。参数thres_steps允许用户传递一个变量以设置衰减率，在这种情况下，
真实的衰减率变为 ：

.. math:: 
    \min(\text{decay}, \frac{1 + \text{thres_steps}}{10 + \text{thres_steps}})

通常thres_steps可以是全局训练steps。
     

参数：
    - **decay** (float) – 指数衰减率，通常接近1，如0.999，0.9999，……
    - **thres_steps** (Variable|None) – 如果不为None，指定衰减率。
    - **name** (str|None) – 名字前缀（可选项）。

**代码示例**

.. code-block:: python

    import numpy
    import paddle
    import paddle.fluid as fluid

    data = fluid.layers.data(name='x', shape=[5], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    cost = fluid.layers.mean(hidden)

    test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(cost)

    global_steps = fluid.layers.learning_rate_scheduler._decay_step_counter()
    ema = fluid.optimizer.ExponentialMovingAverage(0.999, thres_steps=global_steps)
    ema.update()

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    for pass_id in range(3):
        for batch_id in range(6):
            data = numpy.random.random(size=(10, 5)).astype('float32')
            exe.run(program=fluid.default_main_program(),
                feed={'x': data},
                fetch_list=[cost.name])

        # usage 1
        with ema.apply(exe):
            data = numpy.random.random(size=(10, 5)).astype('float32')
            exe.run(program=test_program,
                    feed={'x': data},
                    fetch_list=[hidden.name])


         # usage 2
        with ema.apply(exe, need_restore=False):
            data = numpy.random.random(size=(10, 5)).astype('float32')
            exe.run(program=test_program,
                    feed={'x': data},
                    fetch_list=[hidden.name])
        ema.restore(exe)


.. py:method:: update()

更新指数滑动平均。仅在训练程序中调用此方法。

.. py:method:: apply(executor, need_restore=True)

参数：
    - **executor** (Executor) – 执行应用的执行引擎。
    - **need_restore** (bool) –是否在应用后恢复参数。

.. py:method:: restore(executor)

参数：
    - **executor** (Executor) – 执行存储的执行引擎。




.. _cn_api_fluid_optimizer_Ftrl:

Ftrl
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.Ftrl

``FtrlOptimizer`` 的别名




.. _cn_api_fluid_optimizer_FtrlOptimizer:

FtrlOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.FtrlOptimizer(learning_rate, l1=0.0, l2=0.0, lr_power=-0.5,regularization=None, name=None)
 
FTRL (Follow The Regularized Leader) Optimizer.

FTRL 原始论文: ( `https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf <https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf>`_)


.. math::
           &\qquad new\_accum=squared\_accum+grad^2\\\\
           &\qquad if(lr\_power==−0.5):\\
           &\qquad \qquad linear\_accum+=grad-\frac{\sqrt{new\_accum}-\sqrt{squared\_accum}}{learning\_rate*param}\\
           &\qquad else:\\
           &\qquad \qquad linear\_accum+=grad-\frac{new\_accum^{-lr\_power}-accum^{-lr\_power}}{learning\_rate*param}\\\\
           &\qquad x=l1*sign(linear\_accum)−linear\_accum\\\\
           &\qquad if(lr\_power==−0.5):\\
           &\qquad \qquad y=\frac{\sqrt{new\_accum}}{learning\_rate}+(2*l2)\\
           &\qquad \qquad pre\_shrink=\frac{x}{y}\\
           &\qquad \qquad param=(abs(linear\_accum)>l1).select(pre\_shrink,0.0)\\
           &\qquad else:\\
           &\qquad \qquad y=\frac{new\_accum^{-lr\_power}}{learning\_rate}+(2*l2)\\
           &\qquad \qquad pre\_shrink=\frac{x}{y}\\
           &\qquad \qquad param=(abs(linear\_accum)>l1).select(pre\_shrink,0.0)\\\\
           &\qquad squared\_accum+=grad^2


参数:
  - **learning_rate** (float|Variable)-全局学习率。
  - **l1** (float) - L1 regularization strength.
  - **l2** (float) - L2 regularization strength.
  - **lr_power** (float) - 学习率降低指数
  - **regularization** - 正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** — 可选的名称前缀

抛出异常：
  - ``ValueError`` - 如果 ``learning_rate`` , ``rho`` ,  ``epsilon`` , ``momentum``  为 None.

**代码示例**

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
    
        ftrl_optimizer = fluid.optimizer.Ftrl(learning_rate=0.1)
        ftrl_optimizer.minimize(avg_cost)
    
        fetch_list = [avg_cost]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)


.. note::
     目前, FtrlOptimizer 不支持 sparse parameter optimization




.. _cn_api_fluid_optimizer_LambOptimizer:

LambOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.LambOptimizer(learning_rate=0.001, lamb_weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-06, regularization=None, name=None)

LAMB（Layer-wise Adaptive Moments optimizer for Batching training）优化器
LAMB优化器旨在不降低准确性的条件下扩大训练的批量大小，支持自适应元素更新和精确的分层校正。 更多信息请参考Reducing BERT Pre-Training Time from 3 Days to 76 Minutes。
参数更新如下：

.. math::

    \begin{align}\begin{aligned}m_t^l & = \beta_1 m_{t - 1}^l + (1 - \beta_1)g_t^l\\v_t^l & = \beta_2 v_{t - 1}^l + (1 - \beta_2)g_t^l \odot g_t^l\\\widehat{m}_t^l & = m_t^l/(1 - \beta_1^t)\\\widehat{v}_t^l & = v_t^l/(1 - \beta_2^t)\\r_1 & = \left \| w_{t-1}^l \right \|_2\\r_2 & = \left \|  \frac{\widehat{m}_t^l}{\sqrt{\widehat{v}_t^l+\epsilon}} + \lambda w_{t-1}^l \right \|_2\\r & = r_1 / r_2\\\eta^l & = r \times \eta\\w_t^l & = w_{t-1}^l -\eta ^l \times (\frac{\widehat{m}_t^l}{\sqrt{\widehat{v}_t^l+\epsilon}} + \lambda w_{t-1}^l)\end{aligned}\end{align}

其中 :math:`m` 为第一个时刻，:math:`v` 为第二个时刻，:math:`\eta` 为学习率，:math:`\lambda` 为LAMB权重衰减率。

参数：
    - **learning_rate** (float|Variable) – 用于更新参数的学习速率。可以是浮点值或具有一个作为数据元素的浮点值的变量。
    - **lamb_weight_decay** (float) – LAMB权重衰减率。
    - **beta1** (float) – 第一个时刻估计的指数衰减率。
    - **beta2** (float) – 第二个时刻估计的指数衰减率。
    - **epsilon** (float) – 一个小的浮点值，目的是维持数值稳定性。
    - **regularization** – 一个正则化器，如fluid.regularizer.L1DecayRegularizer。
    - **name** (str|None) – 名字前缀（可选项）。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
     
    data = fluid.layers.data(name='x', shape=[5], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    cost = fluid.layers.mean(hidden)
     
    optimizer = fluid.optimizer.Lamb(learning_rate=0.002)
    optimizer.minimize(cost)



.. _cn_api_fluid_optimizer_LarsMomentum:

LarsMomentum
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.LarsMomentum

``fluid.optimizer.LarsMomentumOptimizer`` 的别名





.. _cn_api_fluid_optimizer_LarsMomentumOptimizer:

LarsMomentumOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.LarsMomentumOptimizer(learning_rate, momentum, lars_coeff=0.001, lars_weight_decay=0.0005, regularization=None, name=None)

LARS支持的Momentum优化器

公式作如下更新：

.. math::

  & local\_learning\_rate = learning\_rate * lars\_coeff * \
  \frac{||param||}{||gradient|| + lars\_weight\_decay * ||param||}\\
  & velocity = mu * velocity + local\_learning\_rate * (gradient + lars\_weight\_decay * param)\\
  & param = param - velocity

参数：
  - **learning_rate** (float|Variable) - 学习率，用于参数更新。作为数据参数，可以是浮点型值或含有一个浮点型值的变量
  - **momentum** (float) - 动量因子
  - **lars_coeff** (float) - 定义LARS本地学习率的权重
  - **lars_weight_decay** (float) - 使用LARS进行衰减的权重衰减系数
  - **regularization** - 正则化函数，例如 :code:`fluid.regularizer.L2DecayRegularizer`
  - **name** - 名称前缀，可选

**代码示例：**

.. code-block:: python

    optimizer = fluid.optimizer.LarsMomentum(learning_rate=0.2, momentum=0.1, lars_weight_decay=0.001)
    optimizer.minimize(cost)







.. _cn_api_fluid_optimizer_ModelAverage:

ModelAverage
-------------------------------

.. py:class:: paddle.fluid.optimizer.ModelAverage(average_window_rate, min_average_window=10000, max_average_window=10000, regularization=None, name=None)

在滑动窗口中累积参数的平均值。平均结果将保存在临时变量中，通过调用 ``apply()`` 方法可应用于当前模型的参数变量。使用 ``restore()`` 方法恢复当前模型的参数值。

平均窗口的大小由 ``average_window_rate`` ， ``min_average_window`` ， ``max_average_window`` 以及当前更新次数决定。

 
参数:
  - **average_window_rate** – 窗口平均速率
  - **min_average_window** – 平均窗口大小的最小值
  - **max_average_window** – 平均窗口大小的最大值
  - **regularization** – 正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** – 可选的名称前缀

**代码示例**

.. code-block:: python
        
    import paddle.fluid as fluid
    import numpy
     
    # 首先创建执行引擎
    place = fluid.CPUPlace()  # fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
     
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        # 构建net
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        optimizer = fluid.optimizer.Momentum(learning_rate=0.2, momentum=0.1)
        optimizer.minimize(loss)

        # 构建ModelAverage优化器
        model_average = fluid.optimizer.ModelAverage(0.15,
                                          min_average_window=10000,
                                          max_average_window=20000)
        exe.run(startup_program)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        outs = exe.run(program=train_program,
                       feed={'X': x},
                       fetch_list=[loss.name])
       # 应用ModelAverage
        with model_average.apply(exe):
             x = numpy.random.random(size=(10, 1)).astype('float32')
             exe.run(program=train_program,
                    feed={'X': x},
                    fetch_list=[loss.name])


.. py:method:: apply(executor, need_restore=True)

将平均值应用于当前模型的参数。

参数：
    - **executor** (fluid.Executor) – 当前的执行引擎。
    - **need_restore** (bool) – 如果您最后需要实现恢复，将其设为True。默认值True。


.. py:method:: restore(executor)

恢复当前模型的参数值

参数：
    - **executor** (fluid.Executor) – 当前的执行引擎。






.. _cn_api_fluid_optimizer_Momentum:

Momentum
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.Momentum

``MomentumOptimizer`` 的别名



.. _cn_api_fluid_optimizer_MomentumOptimizer:

MomentumOptimizer
-------------------------------

.. py:class::  paddle.fluid.optimizer.MomentumOptimizer(learning_rate, momentum, use_nesterov=False, regularization=None, name=None)

含有速度状态的Simple Momentum 优化器

该优化器含有牛顿动量标志，公式更新如下：

.. math::
    & velocity = mu * velocity + gradient\\
    & if (use\_nesterov):\\
    &\quad   param = param - (gradient + mu * velocity) * learning\_rate\\
    & else:\\&\quad   param = param - learning\_rate * velocity

参数：
    - **learning_rate** (float|Variable) - 学习率，用于参数更新。作为数据参数，可以是浮点型值或含有一个浮点型值的变量
    - **momentum** (float) - 动量因子
    - **use_nesterov** (bool) - 赋能牛顿动量
    - **regularization** - 正则化函数，比如fluid.regularizer.L2DecayRegularizer
    - **name** - 名称前缀（可选）

**代码示例**：

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
        
        moment_optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
        moment_optimizer.minimize(avg_cost)
        
        fetch_list = [avg_cost]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)







.. _cn_api_fluid_optimizer_RMSPropOptimizer:

RMSPropOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.RMSPropOptimizer(learning_rate, rho=0.95, epsilon=1e-06, momentum=0.0, centered=False, regularization=None, name=None)

均方根传播（RMSProp）法是一种未发表的,自适应学习率的方法。原演示幻灯片中提出了RMSProp：[http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf]中的第29张。等式如下所示：

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
    - **rho** （float） - rho是等式中的 :math:`rho` ，默认设置为0.95。
    - **epsilon** （float） - 等式中的epsilon是平滑项，避免被零除，默认设置为1e-6。
    - **momentum** （float） - 方程中的β是动量项，默认设置为0.0。
    - **centered** （bool） - 如果为True，则通过梯度的估计方差,对梯度进行归一化；如果False，则由未centered的第二个moment归一化。将此设置为True有助于模型训练，但会消耗额外计算和内存资源。默认为False。
    - **regularization**  - 正则器项，如 ``fluid.regularizer.L2DecayRegularizer`` 。
    - **name**  - 可选的名称前缀。
    
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










.. _cn_api_fluid_optimizer_SGD:

SGD
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.SGD

``SGDOptimizer`` 的别名






.. _cn_api_fluid_optimizer_SGDOptimizer:

SGDOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.SGDOptimizer(learning_rate, regularization=None, name=None)

随机梯度下降算法的优化器

.. math::
            \\param\_out=param-learning\_rate*grad\\


参数:
  - **learning_rate** (float|Variable) - 用于更新参数的学习率。可以是浮点值，也可以是具有一个浮点值作为数据元素的变量。
  - **regularization** - 一个正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** - 可选的名称前缀。
  
  
**代码示例**
 
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
        
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        sgd_optimizer.minimize(avg_cost)

        fetch_list = [avg_cost]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)









