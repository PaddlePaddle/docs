.. _cn_api_paddle_linalg_qr:

qr
-------------------------------

.. py:function:: paddle.linalg.qr(x, mode="reduced", name=None, *, out=None)


计算一个或一批矩阵的正交三角分解，也称 QR 分解（暂不支持反向）。

记 :math:`X` 为一个矩阵，则计算的结果为 2 个矩阵 :math:`Q` 和 :math:`R`，则满足公式：

.. math::
    X = Q * R

其中，:math:`Q` 是正交矩阵，:math:`R` 是上三角矩阵。


参数
::::::::::::

    - **x** (Tensor)：输入进行正交三角分解的一个或一批方阵，类型为 Tensor。 ``x`` 的形状应为 ``[*, M, N]``，其中 ``*`` 为零或更大的批次维度，数据类型支持 float32， float64。别名 ``input``, ``A``。
    - **mode** (str，可选)：控制正交三角分解的行为，默认是 ``reduced``，假设 ``x`` 形状应为 ``[*, M, N]`` 和 ``K = min(M, N)``：
        如果 ``mode = "reduced"``，则 :math:`Q` 形状为 ``[*, M, K]`` 和 :math:`R` 形状为 ``[*, K, N]``；
        如果 ``mode = "complete"``，则 :math:`Q` 形状为 ``[*, M, M]`` 和 :math:`R` 形状为 ``[*, M, N]``；
        如果 ``mode = "r"``，则**只返回**缩减的 :math:`R`，其形状为 ``[*, K, N]``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **out** (Tensor|tuple，可选) - 指定输出的张量。
        当 ``mode = "r"`` 时，为一个 Tensor 用于存储 R；
        否则为一个 ``(Q, R)`` 张量元组，默认值为 None。

返回
::::::::::::

    - 如果 ``mode = "r"``，返回 Tensor R，即缩减的上三角矩阵。
    - 否则返回 ``QrRetType(Q, R)`` 具名元组：
        - Tensor Q，正交三角分解的 Q 正交矩阵。
        - Tensor R，正交三角分解的 R 上三角矩阵。

代码示例
::::::::::

COPY-FROM: paddle.linalg.qr
