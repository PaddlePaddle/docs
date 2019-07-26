.. _cn_api_fluid_layers_sequence_unpad:

sequence_unpad
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_unpad(x, length, name=None)

**实现Sequence Unpad(去除序列填充值)运算**

该层从给定序列中删除padding（填充值），并且将该序列转变为未填充时的原序列作为该层的输出，并且实际长度可以在输出的LoD信息中取得。

::

    示例：

    给定输入变量 ``x`` :
        x.data = [[ 1.0,  2.0,  3.0,  4.0,  5.0],
                  [ 6.0,  7.0,  8.0,  9.0, 10.0],
                  [11.0, 12.0, 13.0, 14.0, 15.0]],

    其中包含 3 个被填充到长度为5的序列，实际长度由输入变量 ``length`` 指明：

        length.data = [[2], [3], [4]],

    则去填充（unpad）后的输出变量为：

        out.data = [[1.0, 2.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0]]
        out.lod = [[2, 3, 4]]



参数:
  - **x** (Variable) – 输入变量，承载着多个填充后等长的序列
  - **length** (Variable) – 变量，指明去填充后各个序列所具有的实际长度
  - **name** (str|None) – 可选项，该层名称。 若为 None, 将自动命名该层

返回：变量，承载着去填充处理后的序列

返回类型：Variable

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[10, 5], dtype='float32')
    len = fluid.layers.data(name='length', shape=[1], dtype='int64')
    out = fluid.layers.sequence_unpad(x=x, length=len)












