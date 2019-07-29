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




