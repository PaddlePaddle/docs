.. _cn_api_tensor_diagonal:

diagonal
-------------------------------

.. py:function:: paddle.diagonal(x, offset=0, axis1=0, axis2=1, name=None)


根据参数 `offset`、`axis1`、`axis2`，返回输入 `Tensor` 的局部视图。

如果输入是 2D Tensor，则返回对角线元素。

如果输入的维度大于 2D，则返回由对角线元素组成的数组，其中对角线从由 axis1 和 axis2 指定的二维平面中获得。默认由输入的前两维组成获得对角线的 2D 平面。

参数 `offset` 确定从指定的二维平面中获取对角线的位置：

    - 如果 offset = 0，则取主对角线。
    - 如果 offset > 0，则取主对角线右上的对角线。
    - 如果 offset < 0，则取主对角线左下的对角线。

参数
:::::::::
    - **x** (Tensor)：输入 Tensor ，必须至少是二维的。输入数据类型应为 bool、int32、int64、float16、float32、float64。
    - **offset** （int，可选）- 从指定的二维平面中获取对角线的位置，默认值为 0，即主对角线。
    - **axis1** （int，可选）- 获取对角线的二维平面的第一维，默认值为 0。
    - **axis2** （int，可选）- 获取对角线的二维平面的第二维，默认值为 1
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    - Tensor (Tensor)，输入 `Tensor` 在指定二维平面的局部视图，数据类型和输入数据类型一致。


代码示例
:::::::::

COPY-FROM: paddle.diagonal
