.. _cn_api_paddle_qr:

qr
-------------------------------

.. py:function:: paddle.qr(input, some=True, *, out=None)

计算一个或一批矩阵的 QR 分解。

该 API 是 ``paddle.linalg.qr`` 的封装，并提供 ``some`` 参数来控制 QR 分解的返回形式。

参数
::::::::::::

    - **input** (Tensor) - 输入 Tensor，形状应为 ``[*, M, N]``，其中 ``*`` 为零或更大的批次维度。
    - **some** (bool，可选) - 控制 QR 分解的行为。若为 ``True``（默认），返回 reduced 的 Q 和 R 矩阵，
      即 Q 的形状为 ``[*, M, K]``，R 的形状为 ``[*, K, N]``，其中 ``K = min(M, N)``。
      若为 ``False``，返回 complete 的 Q 和 R 矩阵，即 Q 的形状为 ``[*, M, M]``，R 的形状为 ``[*, M, N]``

关键字参数
::::::::::::
    - **out** (tuple[Tensor, Tensor]，可选) - 输出 (Q, R) 的元组。默认值为 None。

返回
::::::::::::
tuple[Tensor, Tensor]，QR 分解后的 (Q, R) 元组。

代码示例
::::::::::::

COPY-FROM: paddle.qr
