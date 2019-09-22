.. _cn_api_fluid_layers_expand:

expand
-------------------------------

.. py:function:: paddle.fluid.layers.expand(x, expand_times, name=None)

该OP会根据参数 ``expand_times`` 对输入 ``x`` 的各维度进行复制。通过参数 ``expand_times`` 来为 ``x`` 的每个维度设置复制次数。 ``x`` 的秩应小于等于6。注意， ``expand_times`` 的大小必须与 ``x`` 的秩相同。以下是一个用例：

::

        输入(x) 是一个形状为[2, 3, 1]的 3-D Tensor :

                [
                   [[1], [2], [3]],
                   [[4], [5], [6]]
                ]

        属性(expand_times):  [1, 2, 2]

        输出(out) 是一个形状为[2, 6, 2]的 3-D Tensor:

                [
                    [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
                ]

参数:
        - **x** （Variable）- 维度最高为6的多维 ``Tensor`` 或 ``LoDTensor``，数据类型为 ``float32``，``float64``，``int32`` 或 ``bool``。
        - **expand_times** （list|tuple|Variable）- 数据类型是 ``int32`` 。如果 ``expand_times`` 的类型是 list 或 tuple，它的元素可以是整数或者形状为[1]的 ``Tensor`` 或 ``LoDTensor``。如果 ``expand_times`` 的类型是 ``Variable``，则是1-D ``Tensor`` 或 ``LoDTensor``。表示 ``x`` 每一个维度被复制的次数。
        - **name** （str，可选）- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`。默认值： ``None``。

返回：维度与输入 ``x`` 相同的 ``Tensor`` 或 ``LoDTensor``，数据类型与 ``x`` 相同。返回值的每个维度的大小等于 ``x`` 的相应维度的大小乘以 ``expand_times`` 给出的相应值。

返回类型：``Variable`` 。

抛出异常：
    - :code:`TypeError`：``expand_times`` 的类型应该是 list、tuple 或 Variable。
    - :code:`ValueError`：``expand_times`` 中的元素不能是负值。


**代码示例**

..  code-block:: python

        import paddle.fluid as fluid

        # example 1:
        data_1 = fluid.layers.fill_constant(shape=[2, 3, 1], dtype='int32', value=0)
        expanded_1 = fluid.layers.expand(data_1, expand_times=[1, 2, 2])
        # the shape of expanded_1 is [2, 6, 2].

        # example 2:
        data_2 = fluid.layers.fill_constant(shape=[12, 14], dtype="int32", value=3)
        expand_times = fluid.layers.fill_constant(shape=[2], dtype="int32", value=4)
        expanded_2 = fluid.layers.expand(data_2, expand_times=expand_times)
        # the shape of expanded_2 is [48, 56].








