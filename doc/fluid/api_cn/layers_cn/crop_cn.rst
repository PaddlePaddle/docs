.. _cn_api_fluid_layers_crop:

crop
-------------------------------

.. py:function:: paddle.fluid.layers.crop(x, shape=None, offsets=None, name=None)

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
  - **x** (Variable): 输入张量。
  - **shape** (Variable|list/tuple of integer) - 输出张量的形状由参数shape指定，它可以是一个变量/整数的列表/整数元组。如果是张量变量，它的秩必须与x相同。该方式适可用于每次迭代时候需要改变输出形状的情况。如果是整数列表/tupe，则其长度必须与x的秩相同
  - **offsets** (Variable|list/tuple of integer|None) - 指定每个维度上的裁剪的偏移量。它可以是一个Variable，或者一个整数list/tupe。如果是一个tensor variable，它的rank必须与x相同，这种方法适用于每次迭代的偏移量（offset）都可能改变的情况。如果是一个整数list/tupe，则长度必须与x的rank的相同，如果shape=None，则每个维度的偏移量为0。
  - **name** (str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名。

返回: 裁剪张量。

返回类型: 变量（Variable）

抛出异常: 如果形状不是列表、元组或变量，抛出ValueError


**代码示例**:

..  code-block:: python
    
    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[3, 5], dtype="float32")
    y = fluid.layers.data(name="y", shape=[2, 3], dtype="float32")
    crop = fluid.layers.crop(x, shape=y)


    ## or
    z = fluid.layers.data(name="z", shape=[3, 5], dtype="float32")
    crop = fluid.layers.crop(z, shape=[2, 3])










