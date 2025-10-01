.. _cn_api_paddle_static_nn_spectral_norm:

spectral_norm
-------------------------------

.. py:function:: paddle.static.nn.spectral_norm(weight, dim=0, power_iters=1, eps=1e-12, name=None)

**Spectral Normalization Layer**

该 OP 用于计算了 fc、conv1d、conv2d、conv3d 层的权重参数的谱正则值，输入权重参数应分别为 2-D, 3-D, 4-D, 5-D Tensor，输出 Tensor 与输入 Tensor shape 相同。谱特征值计算方式如下。

步骤 1：生成形状为[H]的向量 U，以及形状为[W]的向量 V，其中 H 是输入权重 Tensor 的第 ``dim`` 个维度，W 是剩余维度的乘积。

步骤 2： ``power_iters`` 应该是一个正整数，用 U 和 V 迭代计算 ``power_iters`` 轮，迭代步骤如下。

.. math::

    \mathbf{v} &:= \frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}\\
    \mathbf{u} &:= \frac{\mathbf{W}^{T} \mathbf{v}}{\|\mathbf{W}^{T} \mathbf{v}\|_2}

步骤 3：计算 :math:`\sigma(\mathbf{W})` 并特征值值归一化。

.. math::
    \sigma(\mathbf{W}) &= \mathbf{u}^{T} \mathbf{W} \mathbf{v}\\
    \mathbf{W} &= \frac{\mathbf{W}}{\sigma(\mathbf{W})}

论文参考：`Spectral Normalization <https://arxiv.org/abs/1802.05957>`_

参数
:::::::::

    - **weight** (Tensor) - spectral_norm 算子的输入权重 Tensor，可以是 2-D, 3-D, 4-D, 5-D Tensor，它是 fc、conv1d、conv2d、conv3d 层的权重，数据类型为 float32 或 float64。
    - **dim** (int，可选) - 将输入（weight）重塑为矩阵之前应排列到第一个的维度索引，如果 input（weight）是 fc 层的权重，则应设置为 0；如果 input（weight）是 conv 层的权重，则应设置为 1，默认为 0。
    - **power_iters** (int，可选) - 将用于计算 spectral norm 的功率迭代次数，默认值 1。
    - **eps** (float，可选) - epsilon 用于保证计算规范中的数值稳定性，分母会加上 ``eps`` 防止除零，默认 1e-12。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

    Tensor，谱正则化后权重 Tensor，维度和数据类型与输入 ``weight`` 一致。


代码示例
:::::::::

COPY-FROM: paddle.static.nn.spectral_norm
