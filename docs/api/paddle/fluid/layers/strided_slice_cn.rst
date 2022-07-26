.. _cn_api_fluid_layers_strided_slice:

strided_slice
-------------------------------
.. py:function:: paddle.fluid.layers.strided_slice(input, axes, starts, ends, strides)



strided_slice算子。

该OP沿多个轴生成 ``input`` 的切片，与numpy类似：https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html。该OP使用 ``axes`` 、 ``starts`` 和 ``ends`` 属性来指定轴列表中每个轴的起点和终点位置，并使用此信息来对 ``input`` 切片。如果向 ``starts`` 或 ``ends`` 传递负值如 :math:`-i`，则表示该轴的反向第 :math:`i-1` 个位置（这里以0为初始位置）， ``strides`` 表示切片的步长，``strides`` 如果为负数，则按照反方向进行切片。如果传递给 ``starts`` 或 ``ends`` 的值大于n（维度中的元素数目），则表示n。当切片一个未知数量的维度时，建议传入 ``INT_MAX``。 ``axes`` 、 ``starts`` 和 ``ends`` 以及 ``strides`` 四个参数的元素数目必须相等。以下示例将解释切片如何工作：

::

        
        示例1：
                给定：
                     data=[[1,2,3,4],[5,6,7,8],]
                     axes=[0,1]
                     starts=[1,0]
                     ends=[2,3]
                     strides=[1,1]

                则：
                     result=[[5,6,7],]
        示例2：
                给定：
                     data=[[1,2,3,4],[5,6,7,8],]
                     axes=[0,1]
                     starts=[1,3]
                     ends=[2,0]
                     strides=[1,-1]

                则：
                     result=[[8,7,6],] 
        示例3：
                给定：
                     data=[[1,2,3,4],[5,6,7,8],]
                     axes=[0,1]
                     starts=[0,1]
                     ends=[-1,1000]    # 此处-1表示第0维的反向第0个位置，索引值是1。
                     strides =[1,3]
                则：
                     result=[[2],]
                     

参数
::::::::::::

       
        - **input** （Variable）- 多维 ``Tensor`` 或 ``LoDTensor``，数据类型为 ``float32``，``float64``，``int32``，或 ``int64``。
        - **axes** （list|tuple）- 数据类型是 ``int32``。表示进行切片的轴。
        - **starts** （list|tuple|Variable）- 数据类型是 ``int32``。如果 ``starts`` 的类型是 list 或 tuple，它的元素可以是整数或者形状为[1]的 ``Tensor`` 或 ``LoDTensor``。如果 ``starts`` 的类型是 ``Variable``，则是1-D ``Tensor`` 或 ``LoDTensor``。表示在各个轴上切片的起始索引值。
        - **ends** （list|tuple|Variable）- 数据类型是 ``int32``。如果 ``ends`` 的类型是 list 或 tuple，它的元素可以是整数或者形状为[1]的 ``Tensor`` 或 ``LoDTensor``。如果 ``ends`` 的类型是 ``Variable``，则是1-D ``Tensor`` 或 ``LoDTensor``。表示在各个轴上切片的结束索引值。
        - **strides** （list|tuple|Variable）- 数据类型是 ``int32``。如果 ``strides`` 的类型是 list 或 tuple，它的元素可以是整数或者形状为[1]的 ``Tensor`` 或 ``LoDTensor``。如果 ``strides`` 的类型是 ``Variable``，则是1-D ``Tensor`` 或 ``LoDTensor``。表示在各个轴上切片的步长。


返回
::::::::::::
多维 ``Tensor`` 或 ``LoDTensor``，数据类型与 ``input`` 相同。


返回类型
::::::::::::
Variable。

抛出异常
::::::::::::

    - :code:`TypeError`：``starts`` 的类型应该是 list、tuple 或 Variable。
    - :code:`TypeError`：``ends`` 的类型应该是 list、tuple 或 Variable。
    - :code:`TypeError`：``strides`` 的类型应该是 list、tuple 或 Variable。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.strided_slice