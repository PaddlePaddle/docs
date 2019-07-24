.. _cn_api_fluid_layers_sequence_expand_as:

sequence_expand_as
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_expand_as(x, y, name=None)

Sequence Expand As Layer

这一层将根据y的第0级lod扩展输入变量x。当前实现要求输入（Y）的lod层数必须为1，输入（X）的第一维应当和输入（Y）的第0层lod的大小相同，不考虑输入（X）的lod。

以下示例解释sequence_expand如何工作：

::

    * 例1:
    给定一维LoDTensor input(X)
        X.data = [[a], [b], [c], [d]]
        X.dims = [4, 1]
    和 input(Y)
        Y.lod = [[0, 3, 6, 7, 8]]
    ref_level: 0
    得到1级 LoDTensor
        Out.lod =  [[0,            3,              6,  7,  8]]
        Out.data = [[a], [a], [a], [b], [b], [b], [c], [d]]
        Out.dims = [8, 1]

    *例2

    给定一个 input(X)：
        X.data = [[a, b], [c, d], [e, f]]
        X.dims = [3, 2]

    和 input(Y):
        Y.lod = [[0, 2, 3, 6]]
    ref_level: 0

    得到输出张量：

        Out.lod =  [[0,             2,     3,                    6]]
        Out.data = [[a, b], [a, b] [c, d], [e, f], [e, f], [e, f]]
        Out.dims = [6, 2]


参数：
    - **x** (Variable) - 输入变量，类型为Tensor或LoDTensor
    - **y** (Variable) - 输入变量，为LoDTensor
    - **name** (str|None) - 该层名称（可选）。如果设为空，则自动为该层命名

返回：扩展变量，LoDTensor

返回类型：变量（Variable）


**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers

    x = fluid.layers.data(name='x', shape=[10], dtype='float32')
    y = fluid.layers.data(name='y', shape=[10, 20],
                     dtype='float32', lod_level=1)
    out = layers.sequence_expand_as(x=x, y=y)







