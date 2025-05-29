.. _cn_api_paddle_distribution_Dirichlet:

Dirichlet
-------------------------------

.. py:class:: paddle.distribution.Dirichlet(concentration)


狄利克雷分布（Dirichlet distribution）是一类在实数域以正单纯形（standard simplex）为支撑集的高维连续概率分布，是 Beta 分布在高维情形的推广。

对独立同分布（independent and identically distributed, iid）的连续随机变量
:math:`\boldsymbol X \in R_k`，和支撑集 :math:`\boldsymbol X \in (0,1), ||\boldsymbol X|| = 1`，其概率密度函数（pdf）为：

.. math::

    f(\boldsymbol X; \boldsymbol \alpha) = \frac{1}{B(\boldsymbol \alpha)} \prod_{i=1}^{k}x_i^{\alpha_i-1}

其中，:math:`\boldsymbol \alpha = {\alpha_1,...,\alpha_k}, k \ge 2` 是无量纲分布参数，:math:`B(\boldsymbol \alpha)` 是多元 Beta 函数。

.. math::

    B(\boldsymbol \alpha) = \frac{\prod_{i=1}^{k} \Gamma(\alpha_i)}{\Gamma(\alpha_0)}

:math:`\alpha_0=\sum_{i=1}^{k} \alpha_i` 是分布参数的和，:math:`\Gamma(\alpha)` 为
Gamma 函数。

参数
:::::::::

- **concentration** (Tensor) - 浓度参数，即上述公式 :math:`\alpha` 参数。当
  concentration 维度大于 1 时，最后一维表示参数，参数形状
  ``event_shape=concentration.shape[-1:]``，其余维为 Batch 维，
  ``batch_shape=concentration.shape[:-1]`` .


代码示例
:::::::::

COPY-FROM: paddle.distribution.Dirichlet

方法
:::::::::

mean
'''''''''

分布均值。


variance
'''''''''

分布方差。


prob(value)
'''''''''

计算 value 的概率。

**参数**

- **value** (Tensor) - 待计算值。

**返回**

- Tensor: value 的概率。


log_prob(value)
'''''''''

计算 value 的对数概率。

**参数**

- **value** (Tensor) - 待计算值。

**返回**

- Tensor: value 的对数概率。


sample(shape=[])
'''''''''

从 Beta 分布中生成满足特定形状的样本数据。

**参数**

- **shape** (Sequence[int]，可选) - 采样次数，最终生成样本形状为 ``shape+batch_shape+event_shape`` 。

**返回**

- Tensor：样本数据。

entropy()
'''''''''

计算 Beta 分布的信息熵。
