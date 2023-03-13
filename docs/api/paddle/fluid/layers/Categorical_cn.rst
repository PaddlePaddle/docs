.. _cn_api_fluid_layers_Categorical:

Categorical
-------------------------------

.. py:class:: paddle.fluid.layers.Categorical(logits)




类别分布是一种离散概率分布，其随机变量可以取 K 个相互独立类别的其中一个。

概率质量函数（pmf）为：

.. math::

    pmf(k; p_i) =\prod_{i=1}^{k} p_i^{[x=i]}

上面公式中：
  - :math:`[x = i]` 表示：如果 :math:`x==i`，则表达式取值为 1，否则取值为 0。


参数
::::::::::::

    - **logits** (list|numpy.ndarray|Variable) - 类别分布对应的 logits。数据类型为 float32。

代码示例
::::::::::::


COPY-FROM: paddle.fluid.layers.Categorical

参数
::::::::::::

    - **other** (Categorical) - 输入的另一个类别分布。数据类型为 float32。

返回
::::::::::::
相对于另一个类别分布的 KL 散度，数据类型为 float32

返回类型
::::::::::::
Variable

.. py:function:: entropy()

信息熵

返回
::::::::::::
类别分布的信息熵，数据类型为 float32

返回类型
::::::::::::
Variable
