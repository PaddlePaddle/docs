.. _cn_api_paddle_distribution_Multinomial:

Multinomial
-------------------------------

.. py:class:: paddle.distribution.Multinomial(total_count, probs)

``Multinomial`` 表示实验次数为 :attr:`total_count`，概率为 :attr:`probs` 的多项分布。

在概率论中，多项分布是二项分布的多元推广，表示具有 :math:`k` 个类别的事件重复实验 :math:`n` 次，每个类别出现次数的概率。当 :math:`k=2` 且 :math:`n=1` 时，为伯努利分布，当 :math:`k=2` 且 :math:`n>1` 时，为二项分布，当 :math:`k>2` 且 :math:`n=1` 时，为分类分布。

多项分布概率密度函数如下：

.. math::

    f(x_1, ..., x_k; n, p_1,...,p_k) = \frac{n!}{x_1!...x_k!}p_1^{x_1}...p_k^{x_k}


其中，:math:`n` 表示实验次数，:math:`k` 表示类别数，:math:`p_i` 表示一次实验中，实验结果为第 :math:`i` 个类别的概率，需要满足 :math:`{\textstyle \sum_{i=1}^{k}p_i=1}, p_i \ge 0` , :math:`x_i` 表示第 :math:`i` 个分类出现的次数。



参数
:::::::::

- **total_count** (int) - 实验次数。
- **probs** (Tensor) - 每个类别发生的概率。最后一维为事件维度，其它维为批维度。:attr:`probs` 中的每个元素取值范围为 :math:`[0, 1]`。如果输入数据大于 1，会沿着最后一维进行归一化操作。

代码示例
:::::::::

COPY-FROM: paddle.distribution.Multinomial


属性
:::::::::

mean
'''''''''

均值

variance
'''''''''

方差


方法
:::::::::

prob(value)
'''''''''

计算 value 的概率。

**参数**

- **value** (Tensor) - 待计算值。

**返回**

Tensor，:attr:`value` 的概率。


log_prob(value)
'''''''''

计算 value 的对数概率。

**参数**

- **value** (Tensor) - 待计算值。

**返回**

Tensor，:attr:`value` 的对数概率。


sample(shape=())
'''''''''

生成满足特定形状的样本数据。

**参数**

- **shape** (Sequence[int]，可选)：采样形状。

**返回**

Tensor，样本数据。
