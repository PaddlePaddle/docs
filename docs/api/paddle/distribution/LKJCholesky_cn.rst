.. _cn_api_paddle_distribution_LKJCholesky:

LKJCholesky
-------------------------------
.. py:class:: paddle.distribution.LKJCholesky(dim, concentration=1.0, sample_method = 'onion')



LKJ 分布是一种用于生成随机相关矩阵的概率分布，广泛应用于贝叶斯统计中，特别是作为协方差矩阵的先验分布。它能够调节相关矩阵的集中度，从而控制变量间的相关性。

LKJ 分布通常定义为对相关矩阵 :math:`\Omega` 的分布，其密度函数为：

.. math::

    p(\Omega \mid \eta) \propto |\Omega|^{\eta - 1}

其中，:math:`\Omega` 是一个 :math:`n \times n` 的相关矩阵，:math:`\eta` 是分布的形状参数，:math:`|\Omega|` 是矩阵的行列式。参数 :math:`\eta` 调节矩阵元素的分布集中度。


相关矩阵的下三角 Choleskey 因子的 LJK 分布支持两种 sample 方法:`onion` 和 `cvine`

参数
::::::::::::

    - **dim** (int) - 目标相关矩阵的维度。
    - **concentration** (float|Tensor) - 集中参数，这个参数控制了生成的相关矩阵的分布，值必须大于 0。concentration 越大，生成的矩阵越接近单位矩阵。
    - **sample_method** (str) - 不同采样策略，可选项有：`onion` 和 `cvine`. 这两种 sample 方法都在 `Generating random correlation matrices based on vines and extended onion method <https://www.sciencedirect.com/science/article/pii/S0047259X09000876>`_ 中提出，并且在相关矩阵上提供相同的分布。但是它们在如何生成样本方面是不同的。默认为“onion”。
代码示例
::::::::::::

COPY-FROM: paddle.distribution.LKJCholesky


方法
:::::::::


log_prob(value)
'''''''''
卡方分布的对数概率密度函数。

**参数**

    - **value** (float|Tensor) - 输入值。

**返回**

    - **Tensor** - value 对应的对数概率密度。



sample(shape=[])
'''''''''
随机采样，生成指定维度的样本。

**参数**

    - **shape** (Sequence[int]，可选) - 采样的样本维度。

**返回**

    - **Tensor** - 指定维度的样本数据。数据类型为 float32。
