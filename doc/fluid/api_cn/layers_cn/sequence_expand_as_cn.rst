.. _cn_api_fluid_layers_sequence_expand_as:

sequence_expand_as
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_expand_as(x, y, name=None)

Sequence Expand As Layer，该OP根据输入 ``y`` 的第0级lod对输入 ``x`` 进行扩展。当前实现要求 ``y`` 的lod层数必须为1，且 ``x`` 的第一维必须和 ``y`` 的第0层lod的大小相同，无需考虑 ``x`` 的lod。

示例如下：

::

    例1:
    给定输入一维LoDTensor x：
        x.data = [[a], [b], [c], [d]]
        x.dims = [4, 1]
    和输入 y：
        y.lod = [[0, 3, 6, 7, 8]]

    输出为1级LoDTensor：
        out.lod =  [[0,            3,              6,  7,  8]]
        out.data = [[a], [a], [a], [b], [b], [b], [c], [d]]
        out.dims = [8, 1]

::

    例2：
    给定输入一维LoDTensor x：
        x.data = [[a, b], [c, d], [e, f]]
        x.dims = [3, 2]
    和输入 y：
        y.lod = [[0, 2, 3, 6]]

    输出为1级LoDTensor：
        out.lod =  [[0,             2,     3,                    6]]
        out.data = [[a, b], [a, b] [c, d], [e, f], [e, f], [e, f]]
        out.dims = [6, 2]


参数：
    - **x** (Variable) - 输入变量，Tensor或LoDTensor。
    - **y** (Variable) - 输入变量，LoDTensor。
    - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回：扩展变量，LoDTensor

返回类型：Variable


**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers

    x = fluid.layers.data(name='x', shape=[10], dtype='float32')
    y = fluid.layers.data(name='y', shape=[10, 20],
                     dtype='float32', lod_level=1)
    out = layers.sequence_expand_as(x=x, y=y)







