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
  - **regularization** (WeightDecayRegularizer, 可选) - 正则化函数，用于减少泛化误差。例如可以是``fluid.regularizer.L2DecayRegularizer``，默认值为None 
  - **epsilon** (float，可选) - 保持数值稳定性的短浮点类型值，默认值为1e-06
  - **name** (str, 可选)- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None

返回：DecayedAdagradOptimizer的实例

返回类型：paddle.fluid.Optimizer

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
    optimizer = fluid.optimizer.DecayedAdagrad(learning_rate=0.2)
    optimizer.minimize(cost)

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

