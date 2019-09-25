.. _cn_api_fluid_layers_crop:

crop
-------------------------------

.. py:function:: paddle.fluid.layers.crop(x, shape=None, offsets=None, name=None)

该OP根据偏移量(offsets)和形状(shape)，裁剪输入张量。

**注意:** 此OP已被弃用，它将在以后的版本中被删除，请使用 :ref:`cn_api_fluid_layers_crop_tensor` 替代

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


参数:
  - **x** (Variable): 多维Tensor，数据类型为float32
  - **shape** (Variable|list/tuple of integers) - 指定输出Tensor的形状，它可以是一个Tensor/整数列表/整数元组。如果是Tensor，它的秩必须与x相同，它的形状指定了输出Tensor的形状，它的元素的数值在这里不起作用，该方式适用于每次迭代时候需要改变输出形状的情况。如果是整数列表/元组，则其长度必须与x的秩相同
  - **offsets** (Variable|list/tuple of integers|None，可选) - 指定每个维度上的裁剪的偏移量，它可以是一个Tensor，或者一个整数列表/整数元组。如果是一个Tensor，它的秩必须与x相同，这种方法适用于每次迭代的偏移量（offset）都可能改变的情况。如果是一个整数列表/元组，则长度必须与x的秩相同，如果offsets=None，则每个维度的偏移量为0。默认值为None
  - **name** (str|None，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回: 经过形状裁剪之后的Tensor，与输入x具有相同的数据类型

返回类型: Variable

抛出异常: 如果形状不是列表、元组或Variable，抛出ValueError


**代码示例**:

..  code-block:: python
    
    import paddle.fluid as fluid
    # case 1
    # 输入x的形状为[-1, 3, 5]，
    # 参数shape = y是个Variable，形状是[-1, 2, 2]，输出Tensor将具有和y一样的形状
    # y的具体数值不起作用，起作用的只有它的形状
    # 经过下面的crop操作之后输出张量的形状是: [-1, 2, 2]
    x = fluid.layers.data(name="x", shape=[3, 5], dtype="float32")
    y = fluid.layers.data(name="y", shape=[2, 2], dtype="float32")
    crop = fluid.layers.crop(x, shape=y)
    ## 或者 case 2
    # 输入z的形状为: [-1, 3, 5], shape为整数列表[-1, 2, 3]
    # 则经过下面的crop操作之后输出张量的形状为：[-1, 2, 3]
    z = fluid.layers.data(name="z", shape=[3, 5], dtype="float32")
    crop = fluid.layers.crop(z, shape=[-1, 2, 3])










