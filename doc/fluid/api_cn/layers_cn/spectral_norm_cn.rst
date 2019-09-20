.. _cn_api_fluid_layers_spectral_norm:

spectral_norm
-------------------------------

.. py:function:: paddle.fluid.layers.spectral_norm(weight, dim=0, power_iters=1, eps=1e-12, name=None)

**Spectral Normalization Layer**

spectral_norm操作用于计算了fc、conv1d、conv2d、conv3d层的权重参数的谱正则值，输入权重参数应分别为2-D, 3-D, 4-D, 5-D张量，输出张量与输入张量shape相同。谱特征值计算方式如下。

步骤1：生成形状为[H]的向量U,以及形状为[W]的向量V,其中H是输入权重张量的第 ``dim`` 个维度，W是剩余维度的乘积。

步骤2： ``power_iters`` 应该是一个正整数，用U和V迭代计算 ``power_iters`` 轮，迭代步骤如下。

.. math::

    \mathbf{v} &:= \frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}\\
    \mathbf{u} &:= \frac{\mathbf{W}^{T} \mathbf{v}}{\|\mathbf{W}^{T} \mathbf{v}\|_2}

步骤3：计算 \sigma(\mathbf{W}) 并特征值值归一化。

.. math::
    \sigma(\mathbf{W}) &= \mathbf{u}^{T} \mathbf{W} \mathbf{v}\\
    \mathbf{W} &= \frac{\mathbf{W}}{\sigma(\mathbf{W})}

可参考: `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_

参数：
    - **weight** (Variable) - spectral_norm算子的输入权重张量，可以是2-D, 3-D, 4-D, 5-D张量，它是fc、conv1d、conv2d、conv3d层的权重。
    - **dim** (int，默认0) - 将输入（weight）重塑为矩阵之前应排列到第一个的维度索引，如果input（weight）是fc层的权重，则应设置为0；如果input（weight）是conv层的权重，则应设置为1，默认为0。
    - **power_iters** (int，默认1) - 将用于计算spectral norm的功率迭代次数，默认值1
    - **eps** (float，默认1e-12) - epsilon用于保证计算规范中的数值稳定性，分母会加上 ``eps`` 防止除零
    - **name** （str|None） - 该层名称（可选）。若设为None，则自动为该层命名。

返回：谱正则化后权重张量

返回类型：Variable

**代码示例**：

.. code-block:: python

   weight = fluid.layers.data(name='weight', shape=[2, 8, 32, 32], append_batch_size=False, dtype='float32')
   x = fluid.layers.spectral_norm(weight=weight, dim=1, power_iters=2)





