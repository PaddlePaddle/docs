.. _cn_api_fluid_layers_slice:

slice
-------------------------------

.. py:function:: paddle.fluid.layers.slice(input, axes, starts, ends)

slice算子。

沿多个轴生成输入张量的切片。与numpy类似： https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html Slice使用 ``axes`` 、 ``starts`` 和 ``ends`` 属性来指定轴列表中每个轴的起点和终点维度，它使用此信息来对输入数据张量切片。如果向 ``starts`` 或 ``ends`` 传递负值，则表示该维度结束之前的元素数目。如果传递给 ``starts`` 或 ``end`` 的值大于n（此维度中的元素数目），则表示n。当切片一个未知数量的唯独时，建议传入INT_MAX. ``axes`` 的大小必须和 ``starts`` 和 ``ends`` 的相等。以下示例将解释切片如何工作：

::

        案例1：
                给定：
                     data=[[1,2,3,4],[5,6,7,8],]
                     axes=[0,1]
                     starts=[1,0]
                     ends=[2,3]
                则：
                     result=[[5,6,7],]

        案例2：
                给定：
                     data=[[1,2,3,4],[5,6,7,8],]
                     starts=[0,1]
                     ends=[-1,1000]
                则：
                     result=[[2,3,4],]

参数：
        - **input** （Variable）- 提取切片的数据张量（Tensor）。
        - **axes** （List）- （list <int>）开始和结束的轴适用于。它是可选的。如果不存在，将被视为[0,1，...，len（starts）- 1]。
        - **starts** （List）- （list <int>）在轴上开始相应轴的索引。
        - **ends** （List）- （list <int>）在轴上结束相应轴的索引。

返回：        切片数据张量（Tensor）.

返回类型：        输出（Variable）。


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid

    starts = [1, 0, 2]
    ends = [3, 3, 4]
    axes = [0, 1, 2]

    input = fluid.layers.data(
        name="input", shape=[3, 4, 5, 6], dtype='float32')

    out = fluid.layers.slice(input, axes=axes, starts=starts, ends=ends)







