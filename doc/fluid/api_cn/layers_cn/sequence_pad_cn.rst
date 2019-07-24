.. _cn_api_fluid_layers_sequence_pad:

sequence_pad
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_pad(x,pad_value,maxlen=None,name=None)

序列填充操作符（Sequence Pad Operator）

这个操作符将同一batch中的序列填充到一个一致的长度。长度由属性padded_length指定。填充的新元素的值具体由输入 ``PadValue`` 指定，并会添加到每一个序列的末尾，使得他们最终的长度保持一致。

以下的例子更清晰地解释此操作符的工作原理：

::

    例1:

    给定 1-level LoDTensor

    input(X):
        X.lod = [[0,2,5]]
        X.data = [a,b,c,d,e]
    input(PadValue):
        PadValue.data = [0]

    'padded_length'=4

    得到LoDTensor:
        Out.data = [[a,b,0,0],[c,d,e,0]]
        Length.data = [[2],[3]]

::

    例2:

    给定 1-level LoDTensor

    input(X):
        X.lod = [[0,2,5]]
        X.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
    input(PadValue):
        PadValue.data = [0]

    'padded_length' = -1,表示用最长输入序列的长度(此例中为3)

    得到LoDTensor:
        Out.data = [[[a1,a2],[b1,b2],[0,0]],[[c1,c2],[d1,d2],[e1,e2]]]
        Length.data = [[2],[3]]


::

    例3:

    给定 1-level LoDTensor

    input(X):
        X.lod = [[0,2,5]]
        X.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
    input(PadValue):
        PadValue.data = [p1,p2]

    'padded_length' = -1,表示用最长输入序列的长度（此例中为3）

    得到LoDTensor:
        Out.data = [[[a1,a2],[b1,b2],[p1,p2]],[[c1,c2],[d1,d2],[e1,e2]]]
        Length.data = [[2],[3]]


参数：
    - **x** (Vairable) - 输入变量，应包含lod信息
    - **pad_value** (Variable) - 变量，存有放入填充步的值。可以是标量或tensor,维度和序列的时间步长相等。如果是标量,则自动广播到时间步长的维度
    - **maxlen** (int,默认None) - 填充序列的长度。可以为空或者任意正整数。当为空时，以序列中最长序列的长度为准，其他所有序列填充至该长度。当是某个特定的正整数，最大长度必须大于最长初始序列的长度
    - **name** (str|None) – 该层的命名(可选项)。 如果为 None, 则自动命名

返回：填充序列批（batch）和填充前的初始长度。所有输出序列的长度相等

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    x = fluid.layers.data(name='y', shape=[10, 5],
                     dtype='float32', lod_level=1)
    pad_value = fluid.layers.assign(
        input=numpy.array([0.0], dtype=numpy.float32))
    out = fluid.layers.sequence_pad(x=x, pad_value=pad_value)









