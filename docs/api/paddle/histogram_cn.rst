.. _cn_api_tensor_histogram:

histogram
-------------------------------

.. py:function:: paddle.histogram(input, bins=100, min=0, max=0):

计算输入张量的直方图。以min和max为range边界，将其均分成bins个直条，然后将排序好的数据划分到各个直条(bins)中。如果min和max都为0, 则利用数据中的最大最小值作为边界。

参数：
    - **input** (Tensor) - 输入Tensor。维度为多维，数据类型为int32, int64, float32或float64。
    - **bins** (int) - 直方图 bins(直条)的个数，默认为100。
    - **min** (int) - range的下边界(包含)，默认为0。
    - **max** (int) - range的上边界(包含)，默认为0。

返回：Tensor，数据为int64类型，维度为(nbins,)。

抛出异常：
    - ``ValueError`` - 当输入 ``bin``, ``min``, ``max``不合法时。

**代码示例**：

.. code-block:: python

    import paddle

    inputs = paddle.to_tensor([1, 2, 1])
    result = paddle.histogram(inputs, bins=4, min=0, max=3)
    print(result) # [0, 2, 1, 0]


