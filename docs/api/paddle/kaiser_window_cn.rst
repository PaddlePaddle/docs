.. _cn_api_paddle_kaiser_window:

kaiser_window
-------------------------------

.. py:function:: paddle.kaiser_window(window_length, periodic=True, beta=12.0, *, dtype='float32', layout=None, device=None, pin_memory=False, requires_grad=False, out=None)

计算 Kaiser 窗。

参数
::::::::::::

    - **window_length** (int) - 返回窗的长度，必须为正数。
    - **periodic** (bool，可选) - 若为 True，则返回适用于周期函数的窗；若为 False，则返回对称窗。默认值为 True。
    - **beta** (float，可选) - 窗函数的形状参数。默认值为 12.0。

关键字参数
::::::::::::

    - **dtype** (str，可选) - 返回 Tensor 的数据类型。默认值为 ``'float32'``。
    - **layout** (str，可选) - 仅为与 PyTorch API 保持一致而保留，在 Paddle 中会被忽略。默认值为 None。
    - **device** (PlaceLike|None，可选) - 返回 Tensor 所在的设备。若为 None，则使用当前设备。默认值为 None。
    - **pin_memory** (bool，可选) - 是否将返回 Tensor 分配在锁页内存中，仅对 CPU Tensor 生效。默认值为 False。
    - **requires_grad** (bool，可选) - 是否为返回 Tensor 记录自动求导。默认值为 False。
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
::::::::::::

Tensor：形状为 ``(window_length,)`` 的一维 Tensor，包含 Kaiser 窗。

代码示例
::::::::::::

COPY-FROM: paddle.kaiser_window
