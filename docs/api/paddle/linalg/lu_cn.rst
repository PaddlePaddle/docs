.. _cn_api_linalg_lu:

lu
-------------------------------

.. py:function:: paddle.linalg.lu(x, pivot=True, get_infos=False, name=None)

对输入的 N 维(N>=2)矩阵 x 进行 LU 分解。

返回 LU 分解矩阵 L、U 和旋转矩阵 P。L 是下三角矩阵，U 是上三角矩阵，拼接成单个矩阵 LU，函数直接返回 LU。

如果 pivot 为 True 则返回旋转矩阵 P 对应序列 pivot，序列 pivot 转换到矩阵 P 可以经如下伪代码实现：

.. code-block:: text

    ones = eye(rows) #eye matrix of rank rows
    for i in range(cols):
        swap(ones[i], ones[pivots[i]])
    return ones

.. note::

    pivot 选项只在 gpu 下起作用，cpu 下暂不支持为 False，会报错。

LU 和 pivot 可以通过调用 paddle.linalg.lu_unpack 展开获得 L、U、P 矩阵。

参数
::::::::::::

    - **x** (Tensor) - 需要进行 LU 分解的输入矩阵 x，x 是维度大于 2 维的矩阵。
    - **pivot** (bool，可选) - LU 分解时是否进行旋转。若为 True 则执行旋转操作，若为 False 则不执行旋转操作，该选项只在 gpu 下起作用，cpu 下暂不支持为 False，会报错。默认 True。
    - **get_infos** (bool，可选) - 是否返回分解状态信息，若为 True，则返回分解状态 Tensor，否则不返回。默认 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - Tensor ``LU``, LU 分解结果矩阵 LU，由 L、U 拼接组成。
    - Tensor(dtype=int) ``Pivots``，旋转矩阵对应的旋转序列，详情见说明部分 pivot 部分，对于输入 ``[*, m, n]`` 的 ``x``，Pivots shape 为 ``[*, m]``。
    - Tensor(dtype=int) ``Infos``，矩阵分解状态信息矩阵，对于输入 ``[*, m, n]``，Infos shape 为 ``[*]``。每个元素表示每组矩阵的 LU 分解是否成功，0 表示分解成功。

代码示例
::::::::::

COPY-FROM: paddle.linalg.lu
