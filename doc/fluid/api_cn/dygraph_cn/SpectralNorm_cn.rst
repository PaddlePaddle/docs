.. _cn_api_fluid_dygraph_SpectralNorm:

SpectralNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.SpectralNorm(name_scope, dim=0, power_iters=1, eps=1e-12, name=None)


该层计算了fc、conv1d、conv2d、conv3d层的权重参数的谱正则值，其参数应分别为2-D, 3-D, 4-D, 5-D。计算结果如下。

步骤1：生成形状为[H]的向量U,以及形状为[W]的向量V,其中H是输入权重的第 ``dim`` 个维度，W是剩余维度的乘积。

步骤2： ``power_iters`` 应该是一个正整数，用U和V迭代计算 ``power_iters`` 轮。

.. math::

    \mathbf{v} &:= \frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}\\
    \mathbf{u} &:= \frac{\mathbf{W}^{T} \mathbf{v}}{\|\mathbf{W}^{T} \mathbf{v}\|_2}

步骤3：计算 \sigma(\mathbf{W}) 并权重值归一化。

.. math::
    \sigma(\mathbf{W}) &= \mathbf{u}^{T} \mathbf{W} \mathbf{v}\\
    \mathbf{W} &= \frac{\mathbf{W}}{\sigma(\mathbf{W})}

可参考: `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_

参数：
    - **name_scope** (str)-该类的名称。
    - **dim** (int)-将输入（weight）重塑为矩阵之前应排列到第一个的维度索引，如果input（weight）是fc层的权重，则应设置为0；如果input（weight）是conv层的权重，则应设置为1，默认为0。
    - **power_iters** (int)-将用于计算spectral norm的功率迭代次数，默认值1。
    - **eps** (float)-epsilon用于计算规范中的数值稳定性，默认值为1e-12
    - **name** (str)-此层的名称，可选。

返回：谱正则化后权重参数的张量变量

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    with fluid.dygraph.guard():
        x = numpy.random.random((2, 8, 32, 32)).astype('float32')
        spectralNorm = fluid.dygraph.nn.SpectralNorm('SpectralNorm', dim=1, power_iters=2)
        ret = spectralNorm(fluid.dygraph.base.to_variable(x))








