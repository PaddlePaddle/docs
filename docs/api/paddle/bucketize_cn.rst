.. _cn_api_paddle_bucketize:

bucketize
-------------------------------

.. py:function:: paddle.bucketize(x, sorted_sequence, out_int32=False, right=False, name=None)
将根据给定的一维 Tensor ``sorted_sequence`` 返回输入 ``x`` 对应的桶索引。

参数
::::::::::
    - **x** (Tensor) - 输入的 N 维 Tensor，支持的数据类型：int32、int64、float32、float64。
    - **sorted_sequence** (Tensor) - 输入的一维 Tensor，支持的数据类型：int32、int64、float32、float64。该 Tensor 的数值在其最后一个维度递增。
    - **out_int32** (bool，可选) - 输出的数据类型支持 int32、int64。默认值为 False，表示默认的输出数据类型为 int64。
    - **right** (bool，可选) - 根据给定 ``x`` 在 ``sorted_sequence`` 查找对应的上边界或下边界。如果 ``sorted_sequence``的值为 nan 或 inf，则返回最内层维度的大小。默认值为 False，表示在 ``sorted_sequence`` 的查找给定 ``x`` 的下边界。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
Tensor(与 ``x`` 维度相同)，如果参数 ``out_int32`` 为 False，则返回数据类型为 int64 的 Tensor，否则将返回 int32 的 Tensor。

代码示例
::::::::::

COPY-FROM: paddle.bucketize
