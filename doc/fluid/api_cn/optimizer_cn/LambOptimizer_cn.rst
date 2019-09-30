.. _cn_api_fluid_optimizer_LambOptimizer:

LambOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.LambOptimizer(learning_rate=0.001, lamb_weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-06, regularization=None, exclude_from_weight_decay_fn=None, name=None)

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
    - **regularization** (Regularizer) – 一个正则化器，如fluid.regularizer.L1DecayRegularizer。
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


.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None)

为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameter_list中的Parameters，最小化网络损失值loss。

参数：
    - **loss** (Variable) – 需要最小化的损失值变量。
    - **startup_program** (Program, 可选) – 用于初始化parameter_list中参数的 :ref:`cn_api_fluid_Program` , 默认值为None，此时将使用 :ref:`cn_api_fluid_default_startup_program` 
    - **parameter_list** (list, 可选) – 待更新的Parameter组成的列表， 默认值为None，此时将更新所有的Parameter
    - **no_grad_set** (set, 可选) – 不需要更新的Parameter的集合，默认值为None
    - **grad_clip** (GradClipBase, 可选) – 梯度裁剪的策略，静态图模式不需要使用本参数，当前本参数只支持在dygraph模式下的梯度裁剪，未来本参数可能会调整，默认值为None

返回： (optimize_ops, params_grads)，数据类型为(list, list)，其中optimize_ops是 ``minimize()`` 接口为网络添加的OP列表，params_grads是一个由(param, grad)变量对组成的列表，param是Parameter，grad是该Parameter对应的梯度值

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








