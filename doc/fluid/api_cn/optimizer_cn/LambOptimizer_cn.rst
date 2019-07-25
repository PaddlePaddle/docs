.. _cn_api_fluid_optimizer_LambOptimizer:

LambOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.LambOptimizer(learning_rate=0.001, lamb_weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-06, regularization=None, exclude_from_weight_decay_fn=None, name=None)

LAMB（Layer-wise Adaptive Moments optimizer for Batching training）优化器
LAMB优化器旨在不降低准确性的条件下扩大训练的批量大小，支持自适应元素更新和精确的分层校正。 更多信息请参考 `Large Batch Optimization for
Deep Learning: Training BERT in 76 minutes <https://arxiv.org/pdf/1904.00962.pdf>`_ 。
参数更新如下：

.. math::

    \begin{align}\begin{aligned}m_t &= \beta_1 m_{t - 1}+ (1 - \beta_1)g_t \\\v_t &= \beta_2 v_{t - 1}  + (1 - \beta_2)g_t^2 \\\r_t &= \frac{m_t}{\sqrt{v_t}+\epsilon} \\\w_t &= w_{t-1} -\eta_t \frac{\left \| w_{t-1}\right \|}{\left \| r_t + \lambda w_{t-1}\right \|} (r_t + \lambda w_{t-1})\end{aligned}\end{align}

其中 :math:`m` 为第一个时刻，:math:`v` 为第二个时刻，:math:`\eta` 为学习率，:math:`\lambda` 为LAMB权重衰减率。

参数：
    - **learning_rate** (float|Variable) – 用于更新参数的学习速率。可以是浮点值或具有一个作为数据元素的浮点值的变量。
    - **lamb_weight_decay** (float) – LAMB权重衰减率。
    - **beta1** (float) – 第一个时刻估计的指数衰减率。
    - **beta2** (float) – 第二个时刻估计的指数衰减率。
    - **epsilon** (float) – 一个小的浮点值，目的是维持数值稳定性。
    - **regularization** (Regularizer) – 一个正则化器，如fluid.regularizer.L1DecayRegularizer。
    - **exclude_from_weight_decay_fn** (function) – 当返回值为True时从权重衰减中去除某个参数。 
    - **name** (str|None) – 名字前缀（可选项）。

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



