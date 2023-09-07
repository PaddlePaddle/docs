.. _cn_api_paddle_eye:

eye
-------------------------------

.. py:function:: paddle.eye(num_rows, num_columns=None, dtype=None, name=None)

构建二维 Tensor(主对角线元素为 1，其他元素为 0)。

参数
::::::::::::

    - **num_rows** (int) - 生成 2-D Tensor 的行数，数据类型为非负 int32。
    - **num_columns** (int，可选) - 生成 2-D Tensor 的列数，数据类型为非负 int32。若为 None，则默认等于 num_rows。
    - **dtype** (np.dtype|str，可选) - 返回 Tensor 的数据类型，可为 float16、float32、float64、int32、int64。若为 None，则默认等于 float32。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 ``shape`` 为 [num_rows, num_columns]的 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.eye
