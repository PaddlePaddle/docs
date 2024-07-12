.. _cn_api_paddle_histogram:

histogram
-------------------------------

.. py:function:: paddle.histogram(input, bins=100, min=0, max=0, weight=None, density=False, name=None)

计算输入 Tensor 的直方图。以 min 和 max 为 range 边界，将其均分成 bins 个直条，然后将排序好的数据划分到各个直条(bins)中。如果 min 和 max 都为 0，则利用数据中的最大最小值作为边界。

参数
::::::::::::

    - **input** (Tensor) - 输入 Tensor。维度为多维，数据类型为 int32、int64、float32 或 float64。
    - **bins** (int，可选) - 直方图 bins(直条)的个数，默认为 100。
    - **min** (int，可选) - range 的下边界(包含)，默认为 0。
    - **max** (int，可选) - range 的上边界(包含)，默认为 0。
    - **weight** (Tensor，可选) - 权重 Tensor，维度和 input 相同。如果提供，输入中的每个值都将以对应的权重值进行计数(而不是 1)。默认为 None。
    - **density** (bool，可选) - 如果为 False，则返回直方图中每个 bin 的计数。如果为 True，则返回直方图中每个 bin 经过归一化后的概率密度函数的值，使得直方图的积分(即所有 bin 的面积)等于 1。默认为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，维度为(nbins,)。如果 density 为 True 或者 weight 不为 None，则返回的 Tensor 的数据类型为 float32；否则为 int64。

代码示例
::::::::::::

COPY-FROM: paddle.histogram
