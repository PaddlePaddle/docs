.. _cn_api_paddle_distribution_Beta:

Beta
-------------------------------

.. py:class:: paddle.distribution.Beta(alpha, beta)


在概率论中，Beta 分布是指一组定义在 [0,1] 区间的连续概率分布，有两个参数
:math:`\alpha,\beta>0`，是狄利克雷(:ref:`cn_api_paddle_distribution_Dirichlet`)
分布的一元形式。

其概率密度函数（pdf）为：

.. math::

    f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)}x^{\alpha-1}(1-x)^{\beta-1}

其中，B 为 Beta 函数，表示归一化因子：

.. math::

  B(\alpha, \beta) = \int_{0}^{1} t^{\alpha - 1} (1-t)^{\beta - 1}\mathrm{d}t

参数
:::::::::

- **alpha** (float|Tensor) - 即上述公式中 :math:`\alpha` 参数，大于零，支持 Broadcast
  语义。当参数类型为 Tensor 时，表示批量创建多个不同参数的分布，``batch_shape`` (参考 :ref:`cn_api_paddle_distribution_Distribution` 基类) 为参数
  Broadcast 后的形状。
- **beta** (float|Tensor) - 即上述公式中 :math:`\beta` 参数，大于零，支持 Broadcast 语
  义。当参数类型为 Tensor 时，表示批量创建多个不同参数的分布，``batch_shape`` (参考 :ref:`cn_api_paddle_distribution_Distribution` 基类) 为参数 Broadcast
  后的形状。

代码示例
:::::::::

COPY-FROM: paddle.distribution.Beta

方法
:::::::::

mean()
'''''''''

计算 Beta 分布均值。


variance()
'''''''''

计算 Beta 分布方差。


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


sample()
'''''''''

从 Beta 分布中生成满足特定形状的样本数据。

**参数**

- **shape** (Sequence[int]，可选)：采样次数。最终生成样本形状为 ``shape+batch_shape`` 。

**返回**

- Tensor：样本数据。

entropy()
'''''''''

计算 Beta 分布的信息熵。
