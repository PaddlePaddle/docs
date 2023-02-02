.. _cn_api_tensor_triu_indices:

triu_indices
--------------------------------

.. py:function:: paddle.triu_indices(row, col=None, offset=0, dtype='int64')

返回行数和列数已知的二维矩阵中上三角矩阵元素的行列坐标，坐标的顺序首先按照行号排列，其次按照列号排列，所述上三角矩阵为原始矩阵某一对角线右上部分元素的子矩阵。

参数
:::::::::
    - **row** (int) - 输入 x 是描述矩阵的行数的一个 int 类型数值。
    - **col** (int，可选) - 输入 x 是描述矩阵的列数的一个 int 类型数值，col 输入默认为 None，此时将 col 设置为 row 的取值，代表输入为正方形矩阵。
    - **offset** (int，可选) - 确定所要考虑的对角线的位置，默认值为 0。

        + 如果 offset = 0，取主对角线。
        + 如果 offset > 0，取主对角线右上的对角线，所包含的元素减少。
        + 如果 offset < 0，取主对角线左下的对角线，所排除的元素减少。

    - **dtype** (str|np.dtype|paddle.dtype，可选) - 指定输出 Tensor 的数据类型，可以是 int32，int64，默认值为 int64。

返回
:::::::::
Tensor，返回 row*col 大小矩阵的上三角元素的坐标，其中第一行包含行坐标，第二行包含列坐标

代码示例
:::::::::

COPY-FROM: paddle.triu_indices
