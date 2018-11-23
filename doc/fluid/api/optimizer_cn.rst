.. _cn_api_fluid_optimizer_MomentumOptimizer:

MomentumOptimizer
>>>>>>>>>>>>>>>>>>

.. py:class::  paddle.fluid.optimizer.MomentumOptimizer(learning_rate, momentum, use_nesterov=False, regularization=None, name=None)

含有速度状态的Simple Momentum 优化器
该优化器含有牛顿动量标志
公式更新如下：

参数：
    - **learning_rate** (float|Variable) - 学习率，用于参数更新。作为数据参数，可以是浮点型值或含有一个浮点型值的变量
    - **momentum** (float) - 动量因子
    - **use_nesterov** (bool) - 赋能牛顿动量
    - **regularization** - 规则化函数，比如fluid.regularizer.L2DecayRegularizer
    - **name** - 名称前缀（可选）

**代码示例**：

.. code_block:: python

    optimizer = fluid.optimizer.Momentum(learning_rate=0.2, momentum=0.1)
    optimizer.minimize(cost)

.. _cn_api_fluid_optimizer_AdagradOptimizer:

AdagradOptimizer
>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.optimizer.AdagradOptimizer(learning_rate, epsilon=1e-06, regularization=None, name=None)

**Adaptive Gradient Algorithm(Adagrad)**

更新如下：

原始论文（http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf）没有epsilon属性。在我们的实现中也作了如下更新：
http://cs231n.github.io/neural-networks-3/#ada 用于维持数值稳定性，避免除数为0的错误发生。

参数：
    - **learning_rate** (float|Variable)-学习率，用于更新参数。作为数据参数，可以是一个浮点类型值或者有一个浮点类型值的变量
    - **epsilon** (float) - 维持数值稳定性的短浮点型值
    - **regularization** - 规则化函数，例如fluid.regularizer.L2DecayRegularizer
    - **name** - 名称前缀（可选）

**代码示例**：

.. code_block:: python:

    optimizer = fluid.optimizer.Adagrad(learning_rate=0.2)
    optimizer.minimize(cost)

AdamOptimizer
>>>>>>>>>>>>>

.. py:class:: paddle.fluid.optimizer. AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, regularization=None, name=None)

该函数实现了自适应矩估计优化器，介绍自Adam论文:https://arxiv.org/abs/1412.6980的第二节。Adam是一阶基于梯度下降的算法，基于自适应低阶矩估计。
Adam更新如下：
    t = t+1
    moment_1_out = 

参数: 
    - **learning_rate** (float|Variable)-学习率，用于更新参数。作为数据参数，可以是一个浮点类型值或有一个浮点类型值的变量
    - **beta1** (float)-一阶矩估计的指数衰减率
    - **beta2** (float)-二阶矩估计的指数衰减率
    - **epsilon** (float)-保持数值稳定性的短浮点类型值
    - **regularization** - 规则化函数，例如''fluid.regularizer.L2DecayRegularizer
    - **name** - 可选名称前缀

**代码示例**：

.. code_block:: python:

    optimizer = fluid.optimizer.Adam(learning_rate=0.2)
    optimizer.minimize(cost)

.. _cn_api_fluid_optimizer_AdamOptimizer:

AdamOptimizer
>>>>>>>>>>>>>

.. py:class:: paddle.fluid.optimizer. AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, regularization=None, name=None)

该函数实现了自适应矩估计优化器，介绍自Adam论文:https://arxiv.org/abs/1412.6980的第二节。Adam是一阶基于梯度下降的算法，基于自适应低阶矩估计。
Adam更新如下：
    t = t+1
    moment_1_out = 

参数: 
    - **learning_rate** (float|Variable)-学习率，用于更新参数。作为数据参数，可以是一个浮点类型值或有一个浮点类型值的变量
    - **beta1** (float)-一阶矩估计的指数衰减率
    - **beta2** (float)-二阶矩估计的指数衰减率
    - **epsilon** (float)-保持数值稳定性的短浮点类型值
    - **regularization** - 规则化函数，例如''fluid.regularizer.L2DecayRegularizer
    - **name** - 可选名称前缀

**代码示例**：

.. code_block:: python:

    optimizer = fluid.optimizer.Adam(learning_rate=0.2)
    optimizer.minimize(cost)