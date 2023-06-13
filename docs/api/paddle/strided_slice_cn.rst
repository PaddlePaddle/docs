.. _cn_api_strided_slice:

strided_slice
-------------------------------
.. py:function:: paddle.strided_slice(x, axes, starts, ends, strides, name)



strided_slice 算子。

沿多个轴生成 ``x`` 的切片，与 numpy 类似：https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html。该 OP 使用 ``axes`` 、 ``starts`` 和 ``ends`` 属性来指定轴列表中每个轴的起点和终点位置，并使用此信息来对 ``x`` 切片。如果向 ``starts`` 或 ``ends`` 传递负值如 :math:`-i`，则表示该轴的反向第 :math:`i-1` 个位置（这里以 0 为初始位置）， ``strides`` 表示切片的步长，``strides`` 如果为负数，则按照反方向进行切片。如果传递给 ``starts`` 或 ``ends`` 的值大于 n（维度中的元素数目），则表示 n。当切片一个未知数量的维度时，建议传入 ``INT_MAX``。 ``axes`` 、 ``starts`` 和 ``ends`` 以及 ``strides`` 四个参数的元素数目必须相等。以下示例将解释切片如何工作：

::


        示例 1：
                给定：
                     data=[[1,2,3,4],[5,6,7,8],]
                     axes=[0,1]
                     starts=[1,0]
                     ends=[2,3]
                     strides=[1,1]

                则：
                     result=[[5,6,7],]
        示例 2：
                给定：
                     data=[[1,2,3,4],[5,6,7,8],]
                     axes=[0,1]
                     starts=[1,3]
                     ends=[2,0]
                     strides=[1,-1]

                则：
                     result=[[8,7,6],]
        示例 3：
                给定：
                     data=[[1,2,3,4],[5,6,7,8],]
                     axes=[0,1]
                     starts=[0,1]
                     ends=[-1,1000]    # 此处-1 表示第 0 维的反向第 0 个位置，索引值是 1。
                     strides =[1,3]
                则：
                     result=[[2],]


参数
::::::::::::


        - **x** （Tensor）- 多维 ``Tensor``，数据类型为 ``bool``, ``float32``，``float64``，``int32``，或 ``int64``。
        - **axes** （list|tuple）- 数据类型是 ``int32``。表示进行切片的轴。
        - **starts** （list|tuple|Tensor）- 数据类型是 ``int32``。如果 ``starts`` 的类型是 list 或 tuple，它的元素可以是整数或者形状为[]的 ``0-D Tensor``。如果 ``starts`` 的类型是 ``Tensor``，则是 1-D ``Tensor``。表示在各个轴上切片的起始索引值。
        - **ends** （list|tuple|Tensor）- 数据类型是 ``int32``。如果 ``ends`` 的类型是 list 或 tuple，它的元素可以是整数或者形状为[]的 ``0-D Tensor``。如果 ``ends`` 的类型是 ``Tensor``，则是 1-D ``Tensor``。表示在各个轴上切片的结束索引值。
        - **strides** （list|tuple|Tensor）- 数据类型是 ``int32``。如果 ``strides`` 的类型是 list 或 tuple，它的元素可以是整数或者形状为[]的 ``0-D Tensor``。如果 ``strides`` 的类型是 ``Tensor``，则是 1-D ``Tensor``。表示在各个轴上切片的步长。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
多维 ``Tensor``，数据类型与 ``x`` 相同。


代码示例
::::::::::::

COPY-FROM: paddle.strided_slice
