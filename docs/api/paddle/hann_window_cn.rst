.. _cn_api_paddle_hann_window:

hann_window
-------------------------------

.. py:function:: paddle.hann_window(
    window_length: int,
    periodic: bool = True,
    *,
    dtype: str = 'float64',
    layout: str | None = None,
    device: PlaceLike | None = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
)

计算一个 Hann 窗。

参数
::::::::::::
    - **window_length** (int): 返回窗口的长度。必须为正数。
    - **periodic** (bool, 可选): 如果为 True，返回一个周期性的窗口；如果为 False，返回一个对称性的窗口。默认值为 True。
    - **dtype** (str, 可选): 返回的 Tensor 的数据类型。默认值为 'float64'。
    - **layout** (str, 可选): 仅用于 API 一致性，Paddle 中忽略。默认值为 None。
    - **device** (PlaceLike|None, 可选): 返回的 Tensor 所在的设备。如果为 None，则使用当前默认设备（参考 paddle.device.set_device()）。对于 CPU Tensor，设备为 CPU；对于 CUDA Tensor，设备为当前 CUDA 设备。默认值为 None。
    - **pin_memory** (bool, 可选): 如果设置为 True，返回的 Tensor 将被分配在固定内存（pinned memory）中。仅对 CPU Tensor 生效。默认值为 False。
    - **requires_grad** (bool, 可选): 是否在 autograd 中记录返回 Tensor 的操作。默认值为 False。


返回
:::::::::

``paddle.Tensor``，一维张量，大小为 `(window_length,)`，表示 Hann 窗口。

代码示例
:::::::::

COPY-FROM: paddle.hann_window
