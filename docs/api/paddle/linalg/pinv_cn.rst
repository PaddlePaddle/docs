.. _cn_api_paddle_linalg_pinv:

pinv
-------------------------------

.. py:function:: paddle.linalg.pinv(x, rcond=1e-15, hermitian=False, name=None, *, atol=None, rtol=None, out=None)

该 API 通过奇异值分解(``svd``)来计算伪逆矩阵，支持单个矩阵或批量矩阵。

    - 如果 ``hermitian`` 为假，那么该 API 会利用奇异值分解(``svd``)进行伪逆矩阵的求解。
    - 如果 ``hermitian`` 为真，那么该 API 会利用特征值分解(``eigh``)进行伪逆矩阵的求解。同时输入需要满足以下条件：如果数据类型为实数，那么输入需要为对称矩阵；如果数据类型为复数，那么输入需要为 ``hermitian`` 矩阵。

参数
::::::::::::
    - **x** (Tensor)：输入变量，类型为 Tensor，数据类型为 float32、float64、complex64、complex128，形状为 ``[..., M, N]``，其中 ``...`` 为零个或多个批次维度，``M`` 和 ``N`` 为任意正整数。当数据类型为 complex64 或 complex128 时，``hermitian`` 必须设为 True。别名 ``input``、``A``。
    - **rcond** (Tensor|float，可选)：用于确定奇异值是否为零的容差值，保留该参数以兼容 NumPy。推荐使用 ``rtol``；若指定 ``rtol``，则忽略 ``rcond``。默认值为 1e-15。
    - **hermitian** (bool，可选)：是否为 ``hermitian`` 矩阵或者实对称矩阵，默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
    - **atol** (float|Tensor|None，可选) - 绝对容差值。为 None 时视为 0。默认值为 None。
    - **rtol** (float|Tensor|None，可选) - 相对容差值。``atol`` 和 ``rtol`` 均为 None 时使用 ``rcond``；若指定 ``rtol``，则忽略 ``rcond``。默认值为 None。
    - **out** (Tensor|None，可选) - 输出 Tensor。若提供，计算结果将写入该 Tensor。默认值为 None。

返回
::::::::::::

Tensor，输入矩阵的伪逆矩阵，数据类型和输入数据类型一致，形状为 ``[..., N, M]``。

代码示例
::::::::::

COPY-FROM: paddle.linalg.pinv
