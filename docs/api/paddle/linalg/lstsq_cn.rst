.. _cn_api_paddle_linalg_lstsq:

lstsq
-------------------------------

.. py:function:: paddle.linalg.lstsq(x, y, rcond=None, driver=None, name=None)


求解线性方程组的最小二乘问题。

参数
::::::::::::

    - **x** (Tensor)：形状为 ``(*, M, N)`` 的矩阵 Tensor， ``*`` 为零或更大的批次维度。数据类型为 float32 或 float64 。
    - **y** (Tensor)：形状为 ``(*, M, K)`` 的矩阵 Tensor， ``*`` 为零或更大的批次维度。数据类型为 float32 或 float64 。
    - **rcond** (float，可选)：默认值为 `None`，用来决定 ``x`` 有效秩的 float 型浮点数。当 ``rcond`` 为 `None` 时，该值会被设为 ``max(M, N)`` 乘 ``x`` 数据类型对应的机器精度。
    - **driver** (str，可选)：默认值为 `None`，用来指定计算使用的 LAPACK 库方法。CPU 下该参数的合法值为 'gels'，'gelsy' (默认)，'gelsd'，'gelss'；CUDA 下该参数的合法值为 'gels' (默认) 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    Tuple，包含 ``solution``、``residuals``、``rank`` 和 ``singular_values``。

    - ``solution`` 指最小二乘解，形状为 ``(*, N, K)`` 的 Tensor。
    - ``residuals`` 指最小二乘解对应的残差，形状为 ``(*, K)`` 的 Tensor；当 ``M > N`` 且 ``x`` 中所有矩阵均为满秩矩阵时，该值会被计算，否则返回空 Tensor。
    - ``rank`` 指 ``x`` 中矩阵的秩，形状为 ``(*)`` 的 Tensor；当 ``driver`` 为 'gelsy', 'gelsd', 'gelss' 时，该值会被计算，否则返回空 Tensor。
    - ``singular_values`` 指 ``x`` 中矩阵的奇异值，形状为 ``(*, min(M, N))`` 的 Tensor；当 ``driver`` 为 'gelsd', 'gelss' 时，该值会被计算，否则返回空 Tensor。

代码示例
::::::::::

COPY-FROM: paddle.linalg.lstsq
