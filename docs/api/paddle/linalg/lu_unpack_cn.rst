.. _cn_api_paddle_linalg_lu_unpack:

lu_unpack
-------------------------------

.. py:function:: paddle.linalg.lu_unpack(x, y, unpack_ludata=True, unpack_pivots=True, name=None)

对 paddle.linalg.lu 返回结果的 LU、pivot 进行展开得到原始的单独矩阵 L、U、P。

从 LU 中获得下三角矩阵 L，上三角矩阵 U。
从序列 pivot 转换得到矩阵 P，其转换过程原理如下伪代码所示：

.. code-block:: text

    ones = eye(rows) #eye matrix of rank rows
    for i in range(cols):
        swap(ones[i], ones[pivots[i]])
    return ones

参数
::::::::::::

    - **x** (Tensor) - paddle.linalg.lu 返回结果的 LU 矩阵。
    - **y** (Tensor) - paddle.linalg.lu 返回结果的 pivot 序列。
    - **unpack_ludata** (bool，可选) - 若为 True，则对输入 x(LU)进行展开得到 L、U，否则。默认 True。
    - **unpack_pivots** (bool，可选) - 若为 True，则对输入 y(pivots)序列进行展开，得到转换矩阵 P。默认 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - Tensor L，由 LU 展开得到的 L 矩阵，若 unpack_ludata 为 False，则为 None。
    - Tensor U，由 LU 展开得到的 U 矩阵，若 unpack_ludata 为 False，则为 None。
    - Tensor P，由序列 pivots 展开得到的旋转矩阵 P，若 unpack_pivots 为 False，则为 None。

代码示例
::::::::::

COPY-FROM: paddle.linalg.lu_unpack
