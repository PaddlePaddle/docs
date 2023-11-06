.. _cn_api_paddle_nn_utils_spectral_norm:

spectral_norm
-------------------------------

.. py:function:: paddle.nn.utils.spectral_norm(layer, name='weight', n_power_iterations=1, eps=1e-12, dim=None)


根据以下步骤对传入的 ``layer`` 中的权重参数进行谱归一化：

步骤 1：生成形状为[H]的向量 U，以及形状为[W]的向量 V，其中 H 是输入权重 Tensor 的第 ``dim`` 个维度，W 是剩余维度的乘积。

步骤 2： ``n_power_iterations`` 是一个正整数，用 U 和 V 迭代计算 ``n_power_iterations`` 轮，迭代步骤如下。

.. math::

    \mathbf{v} &:= \frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}\\
    \mathbf{u} &:= \frac{\mathbf{W} \mathbf{v}}{\|\mathbf{W} \mathbf{v}\|_2}

步骤 3：计算 :math:`\sigma(\mathbf{W})` 并将特征值归一化。

.. math::
    \sigma(\mathbf{W}) &= \mathbf{u}^{T} \mathbf{W} \mathbf{v}\\
    \mathbf{W} &= \frac{\mathbf{W}}{\sigma(\mathbf{W})}

可参考：`Spectral Normalization <https://arxiv.org/abs/1802.05957>`_

参数
::::::::::::

    - **layer** (paddle.nn.Layer) - 要添加权重谱归一化的层。
    - **name** (str，可选) - 权重参数的名字。默认值为 ``weight``。
    - **n_power_iterations** (int，可选) - 将用于计算的 ``SpectralNorm`` 幂迭代次数，默认值：1。
    - **eps** (float，可选) -  ``eps`` 用于保证计算中的数值稳定性，分母会加上 ``eps`` 防止除零。默认值：1e-12。
    - **dim** (int，可选) - 将输入（weight）重塑为矩阵之前应排列到第一个的维度索引，如果 input（weight）是 fc 层的权重，则应设置为 0；如果 input（weight）是 conv 层的权重，则应设置为 1。默认值：None。

返回
::::::::::::

   ``Layer``，添加了权重谱归一化的层

代码示例
::::::::::::

COPY-FROM: paddle.nn.utils.spectral_norm
