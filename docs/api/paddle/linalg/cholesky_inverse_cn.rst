.. _cn_api_paddle_linalg_cholesky_inverse:

cholesky_inverse
-------------------------------

.. py:function:: paddle.linalg.cholesky_inverse(x, upper=False, name=None)

使用 Cholesky 因子 `U` 计算对称正定矩阵的逆矩阵，返回矩阵 `inv`。
如果 `upper` 是 `False`，则 `U` 为下三角矩阵:

    .. math::

        inv = (UU^{T})^{-1}

如果 `upper` 是 `True`，则 `U` 为上三角矩阵:

    .. math::

        inv = (U^{T}U)^{-1}


参数
::::::::::::

    - **x** （Tensor）- 输入变量是形状为 `[N, N]` 的对称正定矩阵的下三角或上三角 Cholesky 分解张量。支持数据类型为 float32、float64。
    - **upper** （bool，可选）- 如果 `upper` 是 `False`，则输入为下三角矩阵，否则为上三角矩阵。默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，求解得到的逆矩阵。

代码示例
::::::::::::

COPY-FROM: paddle.linalg.cholesky_inverse
