.. _cn_api_fluid_layers_sequence_reverse:

sequence_reverse
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_reverse(x, name=None)


在第0维上将输入 ``x`` 的各序列倒序。

::

    假设 ``x`` 是一个形为 (5,4) 的LoDTensor， lod信息为 [[0, 2, 5]]，其中，


    X.data() = [ [1, 2, 3, 4], [5, 6, 7, 8], # 索引为0，长度为2的序列

                 [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20] # 索引为1长度为3的序列

输出 ``Y`` 与 ``x`` 具有同样的维数和LoD信息。 于是有：

::

    Y.data() = [ [5, 6, 7, 8], [1, 2, 3, 4], # 索引为0，长度为2的逆序列
                 [17, 18, 19, 20], [13, 14, 15, 16], [9, 10, 11, 12] # 索引为1，长度为3的逆序列

该运算在建立反dynamic RNN 网络中十分有用。

目前仅支持LoD层次(LoD level)为1的张量倒序。

参数:
  - **x** (Variable) – 输入张量
  - **name** (basestring|None) – 输出变量的命名

返回：输出LoD张量

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[2, 6], dtype='float32')
    x_reversed = fluid.layers.sequence_reverse(x)







