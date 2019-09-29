.. _cn_api_fluid_optimizer_DecayedAdagradOptimizer:

DecayedAdagradOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.DecayedAdagradOptimizer(learning_rate, decay=0.95, epsilon=1e-06, regularization=None, name=None)

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
  - **decay** (float，可选) – 衰减率，默认值为0.95
  - **regularization** (WeightDecayRegularizer, 可选) - 正则化函数，用于减少泛化误差。例如可以是 :ref:`cn_api_fluid_regularizer_L2DecayRegularizer` ，默认值为None 
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

.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None)

为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameter_list中的Parameters，最小化网络损失值loss。

参数：
    - **loss** (Variable) – 需要最小化的损失值变量
    - **startup_program** (Program, 可选) – 用于初始化parameter_list中参数的 :ref:`cn_api_fluid_Program` , 默认值为None，此时将使用 :ref:`cn_api_fluid_default_startup_program` 
    - **parameter_list** (list, 可选) – 待更新的Parameter组成的列表， 默认值为None，此时将更新所有的Parameter
    - **no_grad_set** (set, 可选) – 不需要更新的Parameter的集合，默认值为None
    - **grad_clip** (GradClipBase, 可选) – 梯度裁剪的策略，静态图模式不需要使用本参数，当前本参数只支持在dygraph模式下的梯度裁剪，未来本参数可能会调整，默认值为None

返回： (optimize_ops, params_grads)，数据类型为(list, list)，其中optimize_ops是minimize接口为网络添加的OP列表，params_grads是一个由(param, grad)变量对组成的列表，param是Parameter，grad是该Parameter对应的梯度值

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

