.. _cn_api_fluid_layers_spectral_norm:

spectral_norm
-------------------------------

.. py:function:: paddle.fluid.layers.spectral_norm(weight, dim=0, power_iters=1, eps=1e-12, name=None)

**Spectral Normalization Layer**

该OP用于计算了fc、conv1d、conv2d、conv3d层的权重参数的谱正则值，输入权重参数应分别为2-D, 3-D, 4-D, 5-D张量，输出张量与输入张量shape相同。谱特征值计算方式如下。

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
    - **weight** (Variable) - spectral_norm算子的输入权重张量，可以是2-D, 3-D, 4-D, 5-D Tensor，它是fc、conv1d、conv2d、conv3d层的权重，数据类型为float32或float64。
    - **dim** (int) - 将输入（weight）重塑为矩阵之前应排列到第一个的维度索引，如果input（weight）是fc层的权重，则应设置为0；如果input（weight）是conv层的权重，则应设置为1，默认为0。
    - **power_iters** (int) - 将用于计算spectral norm的功率迭代次数，默认值1
    - **eps** (float) - epsilon用于保证计算规范中的数值稳定性，分母会加上 ``eps`` 防止除零，默认1e-12
    - **name** (None|str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：Variable(Tensor) 谱正则化后权重张量，维度与输入 ``weight`` 一致。

返回类型：变量(Variable)，数据类型与输入 ``weight`` 一致。

**代码示例**：

.. code-block:: python

   weight = fluid.layers.data(name='weight', shape=[2, 8, 32, 32], append_batch_size=False, dtype='float32')
   x = fluid.layers.spectral_norm(weight=weight, dim=1, power_iters=2)





