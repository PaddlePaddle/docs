.. _cn_api_tensor_tril:

tril
-------------------------------

.. py:function:: paddle.tril(input, diagonal=0, name=None)




返回输入矩阵 ``input`` 的下三角部分，其余部分被设为 0。
矩形的下三角部分被定义为对角线上和下方的元素。

参数
:::::::::
    - **input** (Tensor) - 输入 Tensor input，数据类型支持 bool、float32、float64、int32、int64。
    - **diagonal** (int，可选) - 指定的对角线，默认值为 0。如果 diagonal = 0，表示主对角线；如果 diagonal 是正数，表示主对角线之上的对角线；如果 diagonal 是负数，表示主对角线之下的对角线。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor，数据类型与输入 `input` 数据类型一致。


代码示例
:::::::::

COPY-FROM: paddle.tril
