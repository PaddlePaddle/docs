.. _cn_api_paddle_distribution_Bernoulli:

Bernoulli
-------------------------------

.. py:class:: paddle.distribution.Bernoulli(probs, name=None)

伯努利分布由取值为 1 的概率 probs 参数进行初始化。

在概率论和统计学中，以瑞士数学家雅各布·伯努利命名的伯努利分布是随机变量的离散概率分布，其取值 1 的概率为 :math:`p`，值为 0 的概率为 :math:`q = 1 - p` 。

该分布在可能的结果 k 上的概率质量函数为:

.. math::

    {\begin{cases}
    q=1-p & \text{if }value=0 \\
    p & \text{if }value=1
    \end{cases}}

参数
::::::::::::

    - **probs** (float|Tensor) - 伯努利分布的概率输入。数据类型为 float32 或 float64。范围必须为 :math:`[0, 1]`。
    - **name** (str，可选) - 操作的名称，一般无需设置，默认值为 None，具体用法请参见 :ref:`api_guide_Name`。

代码示例
::::::::::::

COPY-FROM: paddle.distribution.Bernoulli

属性
:::::::::

mean
'''''''''

伯努利分布的均值

**返回**

Tensor，均值

variance
'''''''''

伯努利分布的方差

**返回**

Tensor，方差

方法
:::::::::

sample(shape=[])
'''''''''

生成指定维度的样本。

**参数**

    - **shape** (Sequence[int]，可选) - 指定生成样本的维度。

**返回**

Tensor，样本，其维度为 :math:`\text{sample shape} + \text{batch shape} + \text{event shape}`。

**代码示例**

COPY-FROM: paddle.distribution.Bernoulli.sample

rsample(shape=[], temperature=1.0)
'''''''''

重参数化采样，生成指定维度的样本。

``rsample`` 是连续近似的伯努利分布重参数化样本方法。

[1] Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables. 2016.

[2] Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with Gumbel-Softmax. 2016.

.. note::
   ``rsample`` 后面需要跟一个 ``sigmoid``，从而将样本的值转换为单位间隔 :math:`(0, 1)`。

**参数**

    - **shape** (Sequence[int]，可选) - 指定生成样本的维度。
    - **temperature** (float) - ``rsample`` 的温度，必须为正值。

**返回**

Tensor，样本，其维度为 :math:`\text{sample shape} + \text{batch shape} + \text{event shape}`。

**代码示例**

COPY-FROM: paddle.distribution.Bernoulli.rsample

cdf(value)
'''''''''

``value`` 的累积分布函数 （CDF）

.. math::

    { \begin{cases}
    0 & \text{if } value \lt  0 \\
    1 - p & \text{if } 0 \leq value \lt  1 \\
    1 & \text{if } value \geq 1
    \end{cases}
    }

**参数**

    - **value** (Tensor) - 输入 Tensor。

**返回**

Tensor， ``value`` 的累积分布函数。

**代码示例**

COPY-FROM: paddle.distribution.Bernoulli.cdf

log_prob(value)
'''''''''

对数概率密度函数

**参数**

    - **value** (Tensor) - 输入 Tensor。

**返回**

Tensor， ``value`` 的对数概率密度函数。

**代码示例**

COPY-FROM: paddle.distribution.Bernoulli.log_prob

prob(value)
'''''''''

``value`` 的概率密度函数。

.. math::

    { \begin{cases}
        q=1-p & \text{if }value=0 \\
        p & \text{if }value=1
        \end{cases}
    }

**参数**

    - **value** (Tensor) - 输入 Tensor。

**返回**

Tensor， ``value`` 的概率密度函数。

**代码示例**

COPY-FROM: paddle.distribution.Bernoulli.prob

entropy()
'''''''''

伯努利分布的信息熵。

.. math::

    {
        entropy = -(q \log q + p \log p)
    }

**返回**

Tensor，伯努利分布的信息熵。

**代码示例**

COPY-FROM: paddle.distribution.Bernoulli.entropy

kl_divergence(other)
'''''''''

两个伯努利分布之间的 KL 散度。

.. math::

    {
        KL(a || b) = p_a \log(p_a / p_b) + (1 - p_a) \log((1 - p_a) / (1 - p_b))
    }

**参数**

    - **other** (Bernoulli) - ``Bernoulli`` 的实例。

**返回**

Tensor，两个伯努利分布之间的 KL 散度。

**代码示例**

COPY-FROM: paddle.distribution.Bernoulli.kl_divergence
