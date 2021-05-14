.. _cn_api_fluid_layers_Categorical:

Categorical
-------------------------------

.. py:class:: paddle.fluid.layers.Categorical(logits)




类别分布是一种离散概率分布，其随机变量可以取K个相互独立类别的其中一个。

概率质量函数（pmf）为：

.. math::

    pmf(k; p_i) =\prod_{i=1}^{k} p_i^{[x=i]}

上面公式中:
  - :math:`[x = i]` 表示：如果 :math:`x==i` ，则表达式取值为1，否则取值为0。


参数：
    - **logits** (list|numpy.ndarray|Variable) - 类别分布对应的logits。数据类型为float32。

**代码示例**：

.. code-block:: python

    import numpy as np
    from paddle.fluid import layers
    from paddle.fluid.layers import Categorical

    a_logits_npdata = np.array([-0.602,-0.602], dtype="float32")
    a_logits_tensor = layers.create_tensor(dtype="float32")
    layers.assign(a_logits_npdata, a_logits_tensor)

    b_logits_npdata = np.array([-0.102,-0.112], dtype="float32")
    b_logits_tensor = layers.create_tensor(dtype="float32")
    layers.assign(b_logits_npdata, b_logits_tensor)
    
    a = Categorical(a_logits_tensor)
    b = Categorical(b_logits_tensor)

    a.entropy()
    # [0.6931472] with shape: [1]

    b.entropy()
    # [0.6931347] with shape: [1]

    a.kl_divergence(b)
    # [1.2516975e-05] with shape: [1]


.. py:function:: kl_divergence(other)

相对于另一个类别分布的KL散度

参数：
    - **other** (Categorical) - 输入的另一个类别分布。数据类型为float32。
    
返回：相对于另一个类别分布的KL散度, 数据类型为float32

返回类型：Variable

.. py:function:: entropy()

信息熵
    
返回：类别分布的信息熵, 数据类型为float32

返回类型：Variable







