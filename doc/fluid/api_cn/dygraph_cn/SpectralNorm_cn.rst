.. _cn_api_fluid_dygraph_SpectralNorm:

SpectralNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.SpectralNorm(name_scope, dim=0, power_iters=1, eps=1e-12, name=None)

该OP用于计算fc、conv1d、conv2d、conv3d层的权重参数的谱正则值，输入权重参数应分别为2-D, 3-D, 4-D, 5-D张量，输出张量与输入张量维度相同。谱特征值计算方式如下：

步骤1：生成形状为[H]的向量U,以及形状为[W]的向量V,其中H是输入权重张量的第 ``dim`` 个维度，W是剩余维度的乘积。

步骤2： ``power_iters`` 应该是一个正整数，用U和V迭代计算 ``power_iters`` 轮，迭代步骤如下。

.. math::

    \mathbf{v} &:= \frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}\\
    \mathbf{u} &:= \frac{\mathbf{W}^{T} \mathbf{v}}{\|\mathbf{W}^{T} \mathbf{v}\|_2}

步骤3：计算 :math:`\sigma(\mathbf{W})` 并特征值值归一化。

.. math::
    \sigma(\mathbf{W}) &= \mathbf{u}^{T} \mathbf{W} \mathbf{v}\\
    \mathbf{W} &= \frac{\mathbf{W}}{\sigma(\mathbf{W})}

可参考: `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_

参数：
    - **name_scope** (str) - 该类的名称。
    - **dim** (int, 可选) - 将输入（weight）重塑为矩阵之前应排列到第一个的维度索引，如果input（weight）是fc层的权重，则应设置为0；如果input（weight）是conv层的权重，则应设置为1。默认值：0。
    - **power_iters** (int, 可选) - 将用于计算的 ``SpectralNorm`` 功率迭代次数，默认值：1。
    - **eps** (float, 可选) -  ``eps`` 用于保证计算规范中的数值稳定性，分母会加上 ``eps`` 防止除零。默认值：1e-12。
    - **name** (str, 可选) - 层的名称。默认值：None。

返回：无

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    with fluid.dygraph.guard():
        x = np.random.random((2, 8, 32, 32)).astype('float32')
        spectralNorm = fluid.dygraph.nn.SpectralNorm('SpectralNorm', dim=1, power_iters=2)
        ret = spectralNorm(fluid.dygraph.base.to_variable(x))

