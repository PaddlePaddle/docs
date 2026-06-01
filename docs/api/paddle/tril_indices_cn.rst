.. _cn_api_paddle_tril_indices:

tril_indices
--------------------------------

.. py:function:: paddle.tril_indices(row, col, offset=0, dtype='int64', *, device=None)

返回行数和列数已知的二维矩阵中下三角矩阵元素的行列坐标，其中下三角矩阵为原始矩阵某一对角线左下部分元素的子矩阵。

参数
:::::::::
    - **row** (int) - 输入矩阵的行数。
    - **col** (int) - 输入矩阵的列数。
    - **offset** (int，可选) - 确定从指定二维平面中获取对角线的位置。

        + 如果 offset = 0，取主对角线。
        + 如果 offset > 0，取主对角线右上的对角线。
        + 如果 offset < 0，取主对角线左下的对角线。

    - **dtype** (int，可选) - 指定输出 Tensor 的数据类型，默认值为 int64。

关键字参数
:::::::::
    - **device** (str|Place，可选) - 输出 Tensor 的设备位置，可以为字符串或 Place 对象。默认值为 None，表示使用当前设备。

返回
:::::::::
Tensor，二维矩阵的下三角矩阵行坐标和列坐标。数据类型和参数 dtype 一致。

代码示例
:::::::::

COPY-FROM: paddle.tril_indices
