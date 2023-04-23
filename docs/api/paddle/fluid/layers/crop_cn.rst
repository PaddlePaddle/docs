.. _cn_api_fluid_layers_crop:

crop
-------------------------------

.. py:function:: paddle.fluid.layers.crop(x, shape=None, offsets=None, name=None)




该 OP 根据偏移量(offsets)和形状(shape)，裁剪输入 Tensor。

**注意：** 此 OP 已被弃用，它将在以后的版本中被删除，请使用 :ref:`cn_api_fluid_layers_crop_tensor` 替代

**样例**：

::

    * Case 1:
        Given
            X = [[0, 1, 2, 0, 0]
                 [0, 3, 4, 0, 0]
                 [0, 0, 0, 0, 0]],
        and
            shape = [2, 2],
            offsets = [0, 1],
        output is:
            Out = [[1, 2],
                   [3, 4]].
    * Case 2:
        Given
            X = [[0, 1, 2, 5, 0]
                 [0, 3, 4, 6, 0]
                 [0, 0, 0, 0, 0]],
        and shape is tensor
            shape = [[0, 0, 0]
                     [0, 0, 0]]
        and
            offsets = [0, 1],

        output is:
            Out = [[1, 2, 5],
                   [3, 4, 6]].


参数
::::::::::::

  - **x** (Variable)：多维 Tensor，数据类型为 float32
  - **shape** (Variable|list/tuple of integers) - 指定输出 Tensor 的形状，它可以是一个 Tensor/整数列表/整数元组。如果是 Tensor，它的秩必须与 x 相同，它的形状指定了输出 Tensor 的形状，它的元素的数值在这里不起作用，该方式适用于每次迭代时候需要改变输出形状的情况。如果是整数列表/元组，则其长度必须与 x 的秩相同
  - **offsets** (Variable|list/tuple of integers|None，可选) - 指定每个维度上的裁剪的偏移量，它可以是一个 Tensor，或者一个整数列表/整数元组。如果是一个 Tensor，它的秩必须与 x 相同，这种方法适用于每次迭代的偏移量（offset）都可能改变的情况。如果是一个整数列表/元组，则长度必须与 x 的秩相同，如果 offsets=None，则每个维度的偏移量为 0。默认值为 None
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 经过形状裁剪之后的 Tensor，与输入 x 具有相同的数据类型

返回类型
::::::::::::
 Variable

抛出异常
::::::::::::
 如果形状不是列表、元组或 Variable，抛出 ValueError


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.crop
