.. _cn_api_paddle_distribution_MultivariateNormal:

MultivariateNormal
-------------------------------

.. py:class:: paddle.distribution.MultivariateNormal(loc, covariance_matrix=None, precision_matrix=None, scale_tril=None)


MultivariateNormal 是一种定义在实数域上的多元连续型概率分布，参数 :attr:`loc` 表示均值，以及需要传入以下任意一种矩阵描述其方差：
:attr:`covariance_matrix`、 :attr:`precision_matrix`、 :attr:`scale_tril`。

其概率密度函数（pdf）为：

.. math::

    p(X ;\mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} \exp(-\frac{1}{2}(X - \mu)^{\intercal} \Sigma^{-1} (X - \mu))

其中：

- :math:`X` 是 k 维随机向量。
- :math:`\mu` 是 k 维均值向量。
- :math:`\Sigma` 是 k 阶协方差矩阵。


参数
:::::::::

- **loc** (int|float|Tensor) - 即上述公式中 :math:`\mu` 参数，是 MultivariateNormal 的均值向量。如果 :attr:`loc` 的输入数据类型是 `int` 或 `float` 则会被转换为数据类型为 paddle 全局默认数据类型的 1-D Tensor。

- **covariance_matrix** (Tensor) - 即上述公式中 :math:`\mu` 参数，是 MultivariateNormal 的协方差矩阵。:attr:`covariance_matrix` 的数据类型会被转换为与 :attr:`loc` 相同的类型。

- **precision_matrix** (Tensor) - 是 MultivariateNormal 协方差矩阵的逆矩阵。:attr:`precision_matrix` 的数据类型会被转换为与 :attr:`loc` 相同的类型。

- **scale_tril** (Tensor) - 是 MultivariateNormal 协方差矩阵的柯列斯基分解的下三角矩阵。:attr:`scale_tril` 的数据类型会被转换为与 :attr:`loc` 相同的类型。


代码示例
:::::::::

COPY-FROM: paddle.distribution.MultivariateNormal

属性
:::::::::

mean
'''''''''

MultivariateNormal 分布的均值

**返回**

Tensor，均值

variance
'''''''''

MultivariateNormal 分布的方差

**返回**

Tensor，方差

方法
:::::::::

prob(value)
'''''''''

计算 :attr:`value` 的概率。

**参数**

- **value** (Tensor) - 待计算值。

**返回**

- Tensor: :attr:`value` 的概率。数据类型与 `self.loc` 相同。


log_prob(value)
'''''''''

计算 value 的对数概率。

**参数**

- **value** (Tensor) - 待计算值。

**返回**

- Tensor: :attr:`value` 的对数概率。数据类型与 `self.loc` 相同。


sample(shape=())
'''''''''

从 MultivariateNormal 分布中生成满足特定形状的样本数据。最终生成样本形状为 ``sample_shape + batch_shape + event_shape`` 。

**参数**

- **shape** (Sequence[int]，可选)：采样次数。

**返回**

- Tensor：样本数据。其维度为 :math:`\text{sample shape} + \text{batch shape} + \text{event shape}` 。数据类型与 `self.loc` 相同。


rsample(shape=())
'''''''''

重参数化采样，生成指定维度的样本。最终生成样本形状为 ``sample_shape + batch_shape + event_shape`` 。

**参数**

- **shape** (Sequence[int]，可选)：采样次数。

**返回**

- Tensor：样本数据。其维度为 :math:`\text{sample shape} + \text{batch shape} + \text{event shape}` 。数据类型与 `self.loc` 相同。


entropy()
'''''''''

计算 MultivariateNormal 分布的信息熵。

.. math::

    \mathcal{H}(X) = \frac{n}{2} \log(2\pi) + \log {\det A} + \frac{n}{2}

**返回**

类别分布的信息熵，数据类型与 `self.loc` 相同。


kl_divergence(other)
'''''''''

相对于另一个类别分布的 KL 散度，两个分布需要有相同的 :math:`\text{batch shape}` 和 :math:`\text{event shape}`。

.. math::

    KL\_divergence(\lambda_1, \lambda_2) = \log(\det A_2) - \log(\det A_1) -\frac{n}{2} +\frac{1}{2}[tr [\Sigma_2^{-1} \Sigma_1] + (\mu_1 - \mu_2)^{\intercal} \Sigma_2^{-1}  (\mu_1 - \mu_2)]

**参数**

    - **other** (MultivariateNormal) - 输入的另一个类别分布。

**返回**

相对于另一个类别分布的 KL 散度，数据类型与 `self.loc` 相同。
