.. _cn_api_fluid_layers_strided_slice:

strided_slice
-------------------------------
.. py:function:: paddle.fluid.layers.strided_slice(input, axes, starts, ends, strides)
strided_slice算子。

沿多个轴生成输入张量的切片。与numpy类似： https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html。 stried_slice使用 ``axes`` 、 ``starts`` 、 ``ends`` 以及 ``strides`` 属性来指定轴列表中每个轴的起点和终点维度以及步长，它使用此信息来对输入数据张量切片。如果向 ``starts`` 或 ``ends`` 传递负值，则表示该维度结束之前的元素数目。如果传递给 ``starts`` 或 ``end`` 的值大于n（此维度中的元素数目），则表示n。当切片一个未知数量的维度时，建议传入INT_MAX. ``axes`` 的大小必须和 ``starts`` 和 ``ends`` 的相等。以下示例将解释切片如何工作：
该OP沿多个轴生成 ``input`` 的切片。与numpy类似： https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html。该OP使用 ``axes`` 、 ``starts`` 和 ``ends`` 属性来指定轴列表中每个轴的起点和终点位置，并使用此信息来对 ``input`` 切片。如果向 ``starts`` 或 ``ends`` 传递负值如 :math:`-i`，则表示该轴的反向第 :math:`i-1` 个位置（这里以0为初始位置），strrides如果为负数，则按照反方向进行切片。如果传递给 ``starts`` 或 ``end`` 的值大于n（维度中的元素数目），则表示n。当切片一个未知数量的维度时，建议传入 ``INT_MAX``。 ``axes`` 、 ``starts`` 和 ``ends`` 以及 ``strides`` 四个参数的元素数目必须相等。以下示例将解释切片如何工作：

::

        案例1：
        示例1：
                给定：
                     data=[[1,2,3,4],[5,6,7,8],]
                     axes=[0,1]
                     starts=[1,0]
                     ends=[2,3]
                     strides=[1,1]

                则：
                     result=[[5,6,7],]

        案例2：
        示例2：
                给定：
                     data=[[1,2,3,4],[5,6,7,8],]
                     starts=[0,1]
                     ends=[-1,1000]
                     ends=[-1,1000]    # 此处-1表示第0维的反向第0个位置，索引值是1。
                     strides =[1,3]
                则：
                     result=[[2],]
                     

参数：
       
        - **input** （Variable）- 多维 ``Tensor`` 或 ``LoDTensor``，数据类型为 ``float32``，``float64``，``int32``，或 ``int64``。
        - **axes** （list|tuple）- 数据类型是 ``int32``。表示进行切片的轴。它是可选的，如果不存在，将被视为 :math:`[0,1，...，len（starts）- 1]`。
        - **starts** （list|tuple|Variable）- 数据类型是 ``int32``。如果 ``starts`` 的类型是 list 或 tuple，它的元素可以是整数或者形状为[1]的 ``Tensor`` 或 ``LoDTensor``。如果 ``starts`` 的类型是 ``Variable``，则是1-D ``Tensor`` 或 ``LoDTensor``。表示在各个轴上切片的起始索引值。
        - **ends** （list|tuple|Variable）- 数据类型是 ``int32``。如果 ``ends`` 的类型是 list 或 tuple，它的元素可以是整数或者形状为[1]的 ``Tensor`` 或 ``LoDTensor``。如果 ``ends`` 的类型是 ``Variable``，则是1-D ``Tensor`` 或 ``LoDTensor``。表示在各个轴上切片的结束索引值。
        - **strides** （list|tuple|Variable）- 数据类型是 ``int32``。如果 ``strides`` 的类型是 list 或 tuple，它的元素可以是整数或者形状为[1]的 ``Tensor`` 或 ``LoDTensor``。如果 ``ends`` 的类型是 ``Variable``，则是1-D ``Tensor`` 或 ``LoDTensor``。表示在各个轴上切片的步长。

返回：        切片数据张量（Tensor）.
返回：多维 ``Tensor`` 或 ``LoDTensor``，数据类型与 ``input`` 相同。


返回类型：Variable。

抛出异常：
    - :code:`TypeError`：``starts`` 的类型应该是 list、tuple 或 Variable。
    - :code:`TypeError`：``ends`` 的类型应该是 list、tuple 或 Variable。
    - :code:`TypeError`：``strides`` 的类型应该是 list、tuple 或 Variable。

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(
        name="input", shape=[3, 4, 5, 6], dtype='float32')
    # example 1:
    # attr starts is a list which doesn't contain tensor Variable.
    axes = [0, 1, 2]
    starts = [-3, 0, 2]
    ends = [3, 2, 4]
    strides_1 = [1, 1, 1]
    strides = [1, 1, 2]
    sliced_1 = fluid.layers.strided_slice(input, axes=axes, starts=starts, ends=ends, strides=strides_1)
    # sliced_1 is input[:, 0:3:1, 0:2:1, 2:4:1].
    # example 2:
    # attr starts is a list which contain tensor Variable.
    minus_3 = fluid.layers.fill_constant([1], "int32", -3)
    sliced_2 = fluid.layers.slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends, strides=strides_2)
    # sliced_2 is input[:, 0:3:1, 0:2:1, 2:4:2].