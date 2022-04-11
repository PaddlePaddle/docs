.. _cn_api_tensor_tril_indices:

tril_indices
-------------------------------

.. py:function:: paddle.tril_indices(rows, cols, offset=0, dtype=3)


这个op返回行数和列数已知的二维矩阵中下三角形矩阵元素的行列坐标，其中下三角矩阵为原始矩阵某一对角线左下部分元素的子矩阵

参数
:::::::::
    - **rows** (int) - 输入矩阵的行数
    - **cols** (int) - 输入矩阵的列数    
    - **offset** (int，可选): 确定从指定的二维平面中获取对角线的位置
            - 如果 offset = 0，则取主对角线。
            - 如果 offset > 0，则取主对角线右上的对角线。
            - 如果 offset < 0，则取主对角线左下的对角线。
            :math:`\{(i, i)\}` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` 其中
            :math:`d_{1}, d_{2}` 是矩阵行列数.
    - **dtype** (int，可选)- 指定输出张量的数据类型，默认值为int64。

返回
:::::::::
Tensor,二维矩阵的下三角矩阵行坐标和列坐标。数据类型和参数dtype一致。


代码示例
:::::::::

..  code-block:: python

    import numpy as np
    import paddle
            
    # example 1, default offset value
    data1 = paddle.tril_indices(4,4,0)
    print(data1)
    # [[0, 1, 1, 2, 2, 2, 3, 3, 3, 3], 
    #  [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]]

    # example 2, positive offset value
    data2 = paddle.tril_indices(4,4,2)
    print(data2)
    # [[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
    #  [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]]

    # example 3, negative offset value
    data3 = paddle.tril_indices(4,4,-1)
    print(data3)
    # [[ 1, 2, 2, 3, 3, 3],
    #  [ 0, 0, 1, 0, 1, 2]]

