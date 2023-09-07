.. _cn_api_paddle_linalg_pinv:

pinv
-------------------------------

.. py:function:: paddle.linalg.pinv(x, rcond=1e-15, hermitian=False, name=None)

该 API 通过奇异值分解(``svd``)来计算伪逆矩阵，支持单个矩阵或批量矩阵。

    - 如果 ``hermitian`` 为假，那么该 API 会利用奇异值分解(``svd``)进行伪逆矩阵的求解。
    - 如果 ``hermitian`` 为真，那么该 API 会利用特征值分解(``eigh``)进行伪逆矩阵的求解。同时输入需要满足以下条件：如果数据类型为实数，那么输入需要为对称矩阵；如果数据类型为复数，那么输入需要为 ``hermitian`` 矩阵。

参数
:::::::::
    - **x** (Tensor)：输入变量，类型为 Tensor，数据类型为 float32、float64、complex64、complex12，形状为（M, N）或（B, M, N）。
    - **rcond** (float64，可选)：奇异值（特征值）被截断的阈值，奇异值（特征值）小于 rcond * 最大奇异值时会被置为 0，默认值为 1e-15。
    - **hermitian** (bool，可选)：是否为 ``hermitian`` 矩阵或者实对称矩阵，默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor，输入矩阵的伪逆矩阵，数据类型和输入数据类型一致。形状为（N, M）或（B, N, M）。

代码示例
::::::::::

COPY-FROM: paddle.linalg.pinv
