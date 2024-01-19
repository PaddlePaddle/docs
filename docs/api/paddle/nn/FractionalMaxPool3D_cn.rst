.. _cn_api_paddle_nn_FractionalMaxPool3D:


FractionalMaxPool3D
-------------------------------

.. py:function:: paddle.nn.FractionalMaxPool3D(output_size, kernel_size=None, random_u=None, return_mask=False, name=None)

对输入的 Tensor `x` 采取 `2` 维分数阶最大值池化操作，具体可以参考论文：

[1] Ben Graham, Fractional Max-Pooling. 2015. http://arxiv.org/abs/1412.6071

其中输出的 `H` 和 `W` 由参数 `output_size` 决定。

对于各个输出维度，分数阶最大值池化的计算公式为：

..  math::

    \alpha &= size_{input} / size_{output}

    index_{start} &= ceil( \alpha * (i + u) - 1)

    index_{end} &= ceil( \alpha * (i + 1 + u) - 1)

    Output &= max(Input[index_{start}:index_{end}])

    where, u \in (0, 1), i = 0,1,2...size_{output}

公式中的 `u` 即为函数中的参数 `random_u`。另外，由于 `ceil` 对于正小数的操作最小值为 `1` ，因此这里需要再减去 `1` 使索引可以从 `0` 开始计数。

例如，有一个长度为 `7` 的序列 `[2, 4, 3, 1, 5, 2, 3]` ， `output_size` 为 `5` ， `random_u` 为 `0.3`。
则由上述公式可得 `alpha = 7/5 = 1.4` ， 索引的起始序列为 `[0, 1, 3, 4, 6]` ，索引的截止序列为 `[1, 3, 4, 6, 7]` 。
进而得到论文中的随机序列为 `index_end - index_start = [1, 2, 1, 2, 1]` 。
由于池化操作的步长与核尺寸相同，同为此随机序列，最终得到池化输出为 `[2, 4, 1, 5, 3]` 。


参数
:::::::::

    - **output_size** (int|list|tuple)：算子输出图的尺寸，其数据类型为 int 或 list，tuple。如果输出为 tuple 或者 list，则必须包含两个元素， `(H, W)` 。 `H` 和 `W` 可以是 `int` ，也可以是 `None` ，表示与输入保持一致。
    - **kernel_size** (int|list|tuple，可选) - 池化核大小。如果它是一个元组或列表，它必须包含三个整数值，(pool_size_Depth，pool_size_Height, pool_size_Width)。若为一个整数，则表示 D，H 和 W 维度上均为该值，比如若 pool_size=2，则池化核大小为[2,2,2]。默认为 `None`，表示使用 `disjoint` (`non-overlapping`) 模式。
    - **random_u** (float)：分数阶池化操作的浮点随机数，取值范围为 `(0, 1)` 。默认为 `None` ，由框架随机生成，可以使用 `paddle.seed` 设置随机种子。
    - **return_mask** (bool，可选)：如果设置为 `True` ，则会与输出一起返回最大值的索引，默认为 `False`。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 `None`。

形状
:::::::::
    - **x** (Tensor)：默认形状为（批大小，通道数，输出特征深度，高度，宽度），即 NCDHW 格式的 5-D Tensor。其数据类型为 float16, bfloat16, float32, float64。
    - **output** (Tensor)：默认形状为（批大小，通道数，输出特征深度，高度，宽度），即 NCDHW 格式的 5-D Tensor。其数据类型与输入 x 相同。

返回
:::::::::
计算 FractionalMaxPool3D 的可调用对象


代码示例
:::::::::

COPY-FROM: paddle.nn.FractionalMaxPool3D
