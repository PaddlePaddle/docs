.. _cn_api_fluid_layers_sequence_expand_as:

sequence_expand_as
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_expand_as(x, y, name=None)

Sequence Expand As Layer，该OP根据输入 ``y`` 的第0级lod对输入 ``x`` 进行扩展。当前实现要求 ``y`` 的lod层数（level）必须为1，且 ``x`` 的第一维必须和 ``y`` 的第0层lod大小相同，所以扩展后的LodTensor具有和 ``y`` 相同的lod。扩展结果与输入 ``x`` 的lod无关，所以无需考虑 ``x`` 的lod。

注意，该OP的输入 ``x`` 和 ``y`` 只能是LodTensor。

范例解释如下：

::

    例1:
    假设，有4个长度维1的序列[a]、[b]、[c]和[d]，现在要将其扩展为长度是3、4、1、1的序列[a][a][a]、[b][b][b]、[c]和[d]。
    显然，扩展后的序列lod为[0, 3, 6, 7, 8]，则：
    给定输入一维LoDTensor x
        x.data = [[a], [b], [c], [d]]
        x.dims = [4, 1]
    和输入 y
        y.lod = [[0, 3, 6, 7, 8]]
    
    经过sequence_expand_as运算，得到输出1级LoDTensor out
        out.lod =  [[0,            3,              6,  7,  8]]
        out.data = [[a], [a], [a], [b], [b], [b], [c], [d]]
        out.dims = [8, 1]
    
    可见，输出out将x扩展至和y具有相同的lod。

::

    例2：
    设定与例1类似，给定输入一维LoDTensor x：
        x.data = [[a, b], [c, d], [e, f]]
        x.dims = [3, 2]
    和输入 y：
        y.lod = [[0, 2, 3, 6]]

    输出为1级LoDTensor：
        out.lod =  [[0,             2,     3,                    6]]
        out.data = [[a, b], [a, b] [c, d], [e, f], [e, f], [e, f]]
        out.dims = [6, 2]

    可见，输出out将x扩展至和y具有相同的lod。


参数：
    - **x** (Variable) - 输入变量，LoDTensor，第一维必须与输入 ``y`` 的第0层lod大小相同。
    - **y** (Variable) - 输入变量，LoDTensor，lod level必须为1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

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







