.. _cn_api_distribution_Categorical:

Categorical
-------------------------------

.. py:class:: paddle.distribution.Categorical(logits, name=None)




类别分布是一种离散概率分布，其随机变量可以取 K 个相互独立类别的其中一个。

概率质量函数（pmf）为：

.. math::

    pmf(k; p_i) =\prod_{i=1}^{k} p_i^{[x=i]}

上面公式中：

  - :math:`[x = i]` 表示：如果 :math:`x==i`，则表达式取值为 1，否则取值为 0。


参数
::::::::::::

    - **logits** (list|numpy.ndarray|Tensor) - 类别分布对应的 logits。数据类型为 float32 或 float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

代码示例
::::::::::::

COPY-FROM: paddle.distribution.Categorical


方法
:::::::::

sample(shape)
'''''''''

生成指定维度的样本

**参数**

    - **shape** (list) - 指定生成样本的维度。

**返回**

预先设计好维度的 Tensor。

**代码示例**

COPY-FROM: paddle.distribution.Categorical.sample

kl_divergence(other)
'''''''''

相对于另一个类别分布的 KL 散度。

**参数**

    - **other** (Categorical) - 输入的另一个类别分布。数据类型为 float32。

**返回**

相对于另一个类别分布的 KL 散度，数据类型为 float32。

**代码示例**

COPY-FROM: paddle.distribution.Categorical.kl_divergence

entropy()
'''''''''

信息熵。

**返回**

类别分布的信息熵，数据类型为 float32。

**代码示例**

COPY-FROM: paddle.distribution.Categorical.entropy

probs(value)
'''''''''

所选择类别的概率。
如果 ``logits`` 是 2-D 或更高阶的 Tensor，那么其最后一个维度表示不同类别的概率，其它维度被看做不同的概率分布。
同时，如果 ``value`` 是 1-D Tensor，那么 ``value`` 会 broadcast 成与 ``logits`` 具有相同的概率分布数量。
如果 ``value`` 为更高阶 Tensor，那么 ``value`` 应该与 ``logits`` 具有相同的概率分布数量。也就是说，``value.shape[:-1] = logits.shape[:-1]`` 。

**参数**

    - **value** (Tensor) - 输入 Tensor，表示选择的类别下标。数据类型为 int32 或 int64。

**返回**

给定类别下标的概率。

COPY-FROM: paddle.distribution.Categorical.probs

log_prob(value)
'''''''''

所选择类别的对数概率。

**参数**

    - **value** (Tensor) - 输入 Tensor，表示选择的类别下标。数据类型为 int32 或 int64。

**返回**

对数概率。

COPY-FROM: paddle.distribution.Categorical.log_prob
