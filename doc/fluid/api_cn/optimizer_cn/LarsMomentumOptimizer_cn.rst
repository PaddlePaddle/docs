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

    import paddle.fluid as fluid
    optimizer = fluid.optimizer.LarsMomentum(learning_rate=0.2, momentum=0.1, lars_weight_decay=0.001)
    optimizer.minimize(cost)







