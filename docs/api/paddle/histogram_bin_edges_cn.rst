.. _cn_api_paddle_histogram_bin_edges:

histogram_bin_edges
-------------------------------

.. py:function:: paddle.histogram_bin_edges(input, bins=100, min=0, max=0, name=None)

只返回在计算 ``input`` 的直方图时所使用的 bins 的边界值 bin_edges。如果 min 和 max 都为 0，则利用数据中的最大最小值作为边界。

参数
::::::::::::

    - **input** (Tensor) - 输入 Tensor。维度为多维，数据类型为 int32、int64、float32 或 float64。
    - **bins** (int，可选) - 直方图 bins(直条)的个数，默认为 100。
    - **min** (int，可选) - range 的下边界(包含)，默认为 0。
    - **max** (int，可选) - range 的上边界(包含)，默认为 0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，计算直方图时所使用的边界值 bin_edges，返回的数据类型为 float32。

代码示例
::::::::::::

COPY-FROM: paddle.histogram_bin_edges
