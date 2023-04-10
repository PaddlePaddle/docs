.. _cn_api_nn_functional_dropout:

dropout
-------------------------------

.. py:function:: paddle.nn.functional.dropout(x, p=0.5, axis=None, training=True, mode="upscale_in_train", name=None)

Dropout 是一种正则化手段，可根据给定的丢弃概率 `p`，在训练过程中随机将一些神经元输出设置为 0，通过阻止神经元节点间的相关性来减少过拟合。

参数
:::::::::
 - **x** (Tensor) - 输入的多维 `Tensor`，数据类型为：float16、float32、float64。
 - **p** (float，可选) - 将输入节点置 0 的概率，即丢弃概率。默认值为 0.5。
 - **axis** (int|list，可选) - 指定对输入 `Tensor` 进行 dropout 操作的轴。默认值为 None。
 - **training** (bool，可选) - 标记是否为训练阶段。默认值为 True。
 - **mode** (str，可选) - 丢弃单元的方式，有 'upscale_in_train' 和 'downscale_in_infer' 两种可供选择，默认值为 'upscale_in_train'。计算方法如下：

    1. upscale_in_train（默认值），在训练时增大输出结果。

       - 训练时： :math:`out = input \times \frac{mask}{(1.0 - p)}`
       - 预测时： :math:`out = input`

    2. downscale_in_infer，在预测时减小输出结果

       - 训练时： :math:`out = input \times mask`
       - 预测时： :math:`out = input \times (1.0 - p)`

 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
经过 dropout 之后的结果，与输入 x 形状相同的 `Tensor` 。

使用示例 1
:::::::::
axis 参数的默认值为 None。当 ``axis=None`` 时，dropout 的功能为：对输入 Tensor x 中的任意元素，以丢弃概率 p 随机将一些元素输出置 0。这是我们最常见的 dropout 用法。

 -  下面以一个示例来解释它的实现逻辑，同时展示其它参数的含义。

..  code-block:: text

   假定 x 是形状为 2*3 的 2 维 Tensor：
   [[1 2 3]
    [4 5 6]]
   在对 x 做 dropout 时，程序会先生成一个和 x 相同形状的 mask Tensor，mask 中每个元素的值为 0 或 1。
   每个元素的具体值，则是依据丢弃概率从伯努利分布中随机采样得到。
   比如，我们可能得到下面这样一个 2*3 的 mask:
   [[0 1 0]
    [1 0 1]]
   将输入 x 和生成的 mask 点积，就得到了随机丢弃部分元素之后的结果：
   [[0 2 0]
    [4 0 6]]
   假定 dropout 的概率使用默认值，即 ``p=0.5``，若 mode 参数使用默认值，即 ``mode='upscale_in_train'`` ，
   则在训练阶段，最终增大后的结果为：
   [[0 4 0 ]
    [8 0 12]]
   在测试阶段，输出跟输入一致：
   [[1 2 3]
    [4 5 6]]
   若参数 mode 设置为'downscale_in_infer'，则训练阶段的输出为：
   [[0 2 0]
    [4 0 6]]
   在测试阶段，缩小后的输出为：
   [[0.5 1.  1.5]
    [2.  2.5 3. ]]

使用示例 2
:::::::::
若参数 axis 不为 None，dropout 的功能为：以一定的概率从图像特征或语音序列中丢弃掉整个通道。

 -  axis 应设置为：``[0, 1, ... , ndim(x)-1]`` 的子集（ndim(x) 为输入 x 的维度），例如：

   - 若 x 的维度为 2，参数 axis 可能的取值有 4 种：``None``, ``[0]``, ``[1]``, ``[0,1]``
   - 若 x 的维度为 3，参数 axis 可能的取值有 8 种：``None``, ``[0]``, ``[1]``, ``[2]``, ``[0,1]``, ``[0,2]``, ``[1,2]``, ``[0,1,2]``

 -  下面以维度为 2 的输入 Tensor 展示 axis 参数的用法：

..  code-block:: text

   假定 x 是形状为 2*3 的 2 维 Tensor:
   [[1 2 3]
    [4 5 6]]
   (1) 若 ``axis=[0]``，则表示只在第 0 个维度做 dropout。这时生成 mask 的形状为 2*1。
     例如，我们可能会得到这样的 mask:
     [[1]
      [0]]
     这个 2*1 的 mask 在和 x 做点积的时候，会首先广播成一个 2*3 的矩阵：
     [[1 1 1]
      [0 0 0]]
     点积所得的结果为：
     [[1 2 3]
      [0 0 0]]
     之后依据其它参数的设置，得到最终的输出结果。

   (2) 若 ``axis=[1]``，则表示只在第 1 个维度做 dropout。这时生成的 mask 形状为 1*3。
     例如，我们可能会得到这样的 mask:
     [[1 0 1]]
     这个 1*3 的 mask 在和 x 做点积的时候，会首先广播成一个 2*3 的矩阵：
     [[1 0 1]
      [1 0 1]]
     点积所得结果为：
     [[1 0 3]
      [4 0 6]]
   (3) 若 ``axis=[0, 1]``，则表示在第 0 维和第 1 维上做 dropout。此时与默认设置 ``axis=None`` 的作用一致。

若输入 x 为 4 维 Tensor，形状为 `NCHW`，其中 N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度，当设置 ``axis=[0,1]`` 时，则只会在通道 `N` 和 `C` 上做 dropout，通道 `H` 和 `W` 的元素是绑定在一起的，即：``paddle.nn.functional.dropout(x, p, axis=[0,1])``，此时对 4 维 Tensor 中的某个 2 维特征图（形状为 `HW`），或者全部置 0，或者全部保留，这便是 dropout2d 的实现。详情参考 :ref:`cn_api_nn_functional_dropout2d` 。

类似的，若输入 x 为 5 维 Tensor，形状为 `NCDHW`，其中 D 是特征深度，当设置 ``axis=[0,1]`` 时，便可实现 dropout3d。详情参考 :ref:`cn_api_nn_functional_dropout3d` 。

.. note::
   关于广播 (broadcasting) 机制，如您想了解更多，请参见 `Tensor 介绍`_ .

   .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.dropout
