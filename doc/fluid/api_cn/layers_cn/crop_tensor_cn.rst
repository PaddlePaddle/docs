.. _cn_api_fluid_layers_crop_tensor:

crop_tensor
-------------------------------

.. py:function:: paddle.fluid.layers.crop_tensor(x, shape=None, offsets=None, name=None)

根据偏移量（offsets）和形状（shape），裁剪输入张量。

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
                X =  [[[0, 1, 2, 3]
                       [0, 5, 6, 7]
                       [0, 0, 0, 0]],

                      [[0, 3, 4, 5]
                       [0, 6, 7, 8]
                       [0, 0, 0, 0]]].
            and
                shape = [2, 2, 3],
                offsets = [0, 0, 1],
            output is:
                Out = [[[1, 2, 3]
                        [5, 6, 7]],

                       [[3, 4, 5]
                        [6, 7, 8]]].

参数:
  - **x** (Variable): 输入张量。
  - **shape** (Variable|list|tuple of integer) - 输出张量的形状由参数shape指定，它可以是一个1-D的变量/列表/整数元组。如果是1-D的变量，它的秩必须与x相同。如果是列表或整数元组，则其长度必须与x的秩相同。当它是列表时，每一个元素可以是整数或者shape为[1]的变量。含有变量的方式适用于每次迭代时需要改变输出形状的情况。列表和元组中只有第一个元素可以被设置为-1，这意味着输出的第一维大小与输入相同。
  - **offsets** (Variable|list|tuple of integer|None) - 指定每个维度上的裁剪的偏移量。它可以是一个1-D的变量/列表/整数元组。如果是1-D的变量，它的秩必须与x相同。如果是列表或整数元组，则其长度必须与x的秩相同。当它是列表时，每一个元素可以是整数或者shape为[1]的变量。含有变量的方式适用于每次迭代的偏移量（offset）都可能改变的情况。如果offsets=None，则每个维度的偏移量为0。
  - **name** (str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名。

返回: 裁剪张量。

返回类型: 变量（Variable）

抛出异常: 如果形状不是列表、元组或变量，抛出ValueError

抛出异常: 如果偏移量不是None、列表、元组或变量，抛出ValueError

**代码示例**:

..  code-block:: python
    
    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[3, 5], dtype="float32")
    # x.shape = [-1, 3, 5], where -1 indicates batch size, and it will get the exact value in runtime.

    # shape is a 1-D tensor variable
    crop_shape = fluid.layers.data(name="crop_shape", shape=[3], dtype="int32", append_batch_size=False)
    crop0 = fluid.layers.crop_tensor(x, shape=crop_shape)
    # crop0.shape = [-1, -1, -1], it means crop0.shape[0] = x.shape[0] in runtime.

    # or shape is a list in which each element is a constant
    crop1 = fluid.layers.crop_tensor(x, shape=[-1, 2, 3])
    # crop1.shape = [-1, 2, 3]

    # or shape is a list in which each element is a constant or variable
    y = fluid.layers.data(name="y", shape=[3, 8, 8], dtype="float32")
    dim1 = fluid.layers.data(name="dim1", shape=[1], dtype="int32", append_batch_size=False)
    crop2 = fluid.layers.crop_tensor(y, shape=[-1, 3, dim1, 4])
    # crop2.shape = [-1, 3, -1, 4]

    # offsets is a 1-D tensor variable
    crop_offsets = fluid.layers.data(name="crop_offsets", shape=[3], dtype="int32", append_batch_size=False)
    crop3 = fluid.layers.crop_tensor(x, shape=[-1, 2, 3], offsets=crop_offsets)
    # crop3.shape = [-1, 2, 3]

    # offsets is a list in which each element is a constant or variable
    offsets_var =  fluid.layers.data(name="dim1", shape=[1], dtype="int32", append_batch_size=False)
    crop4 = fluid.layers.crop_tensor(x, shape=[-1, 2, 3], offsets=[0, 1, offsets_var])
    # crop4.shape = [-1, 2, 3]









