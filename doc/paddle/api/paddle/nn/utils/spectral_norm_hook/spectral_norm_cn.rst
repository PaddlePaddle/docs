.. __cn_api_nn_cn_spectral_norm:

spectral_norm
-------------------------------

.. py:function:: paddle.nn.utils.spectral_norm(layer, name='weight', n_power_iterations=1, eps=1e-12, dim=None)


该接口根据以下步骤对传入的 ``layer`` 中的权重参数进行谱归一化:

步骤1：生成形状为[H]的向量U,以及形状为[W]的向量V,其中H是输入权重张量的第 ``dim`` 个维度，W是剩余维度的乘积。

步骤2： ``power_iters`` 应该是一个正整数，用U和V迭代计算 ``power_iters`` 轮，迭代步骤如下。

.. math::

    \mathbf{v} &:= \frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}\\
    \mathbf{u} &:= \frac{\mathbf{W} \mathbf{v}}{\|\mathbf{W} \mathbf{v}\|_2}

步骤3：计算 :math:`\sigma(\mathbf{W})` 并将特征值归一化。

.. math::
    \sigma(\mathbf{W}) &= \mathbf{u}^{T} \mathbf{W} \mathbf{v}\\
    \mathbf{W} &= \frac{\mathbf{W}}{\sigma(\mathbf{W})}

可参考: `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_

参数
    - **layer** (paddle.nn.Layer) - 要添加权重谱归一化的层。
    - **name** (str, 可选) - 权重参数的名字。默认：'weight'.
    - **n_power_iterations** (int, 可选) - 将用于计算的 ``SpectralNorm`` 功率迭代次数，默认值：1。
    - **eps** (float, 可选) -  ``eps`` 用于保证计算规范中的数值稳定性，分母会加上 ``eps`` 防止除零。默认值：1e-12。
    - **dim** (int, 可选) - 将输入（weight）重塑为矩阵之前应排列到第一个的维度索引，如果input（weight）是fc层的权重，则应设置为0；如果input（weight）是conv层的权重，则应设置为1。默认值：None。

返回：
   ``Layer`` , 添加了权重谱归一化的层

**代码示例**

.. code-block:: python

    from paddle.nn import Conv2D
    from paddle.nn.utils import Spectralnorm

    conv = Conv2D(3, 1, 3)
    sn_conv = spectral_norm(conv)
    print(sn_conv)
    # Conv2D(3, 1, kernel_size=[3, 3], data_format=NCHW)
    print(sn_conv.weight)
    # Tensor(shape=[1, 3, 3, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
    #        [[[[-0.21090528,  0.18563725, -0.14127982],
    #           [-0.02310637,  0.03197737,  0.34353802],
    #           [-0.17117859,  0.33152047, -0.28408015]],
    # 
    #          [[-0.13336606, -0.01862637,  0.06959272],
    #           [-0.02236020, -0.27091628, -0.24532901],
    #           [ 0.27254242,  0.15516677,  0.09036587]],
    # 
    #          [[ 0.30169338, -0.28146112, -0.11768346],
    #           [-0.45765871, -0.12504843, -0.17482486],
    #           [-0.36866254, -0.19969313,  0.08783543]]]])
