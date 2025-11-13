.. _cn_api_paddle_hamming_window:

hamming_window
-------------------------------

.. py:function:: paddle.hamming_window(
    window_length: int,
    periodic: bool = True,
    alpha: float = 0.54,
    beta: float = 0.46,
    *,
    dtype: str = 'float64',
    layout: str | None = None,
    device: str | None = None,
    pin_memory: None | bool = None,
    requires_grad: bool = False,
)

计算一个广义的汉明窗。

参数
::::::::::::
    - **window_length** (int): 返回窗口的长度。必须为正数。
    - **periodic** (bool, 可选): 如果为 True，返回一个周期性的窗口；如果为 False，返回一个对称性的窗口。默认值为 True。
    - **alpha** (float, 可选): 窗口系数 α。默认值为 0.54。
    - **beta** (float, 可选): 窗口系数 β。默认值为 0.46。
    - **dtype** (str, 可选): 返回的 Tensor 的数据类型。默认值为 'float64'。
    - **layout** (str, 可选): 仅用于 API 一致性，Paddle 中忽略。默认值为 None。
    - **device** (str, 可选): 返回的 Tensor 存放的设备位置。默认值为 None（使用默认设备）。
    - **pin_memory** (bool, 可选): 如果为 True，返回的 Tensor 会分配在 pinned 内存中；否则不会。仅适用于 CPU Tensor。默认值为 None。
    - **requires_grad** (bool, 可选): 如果为 True，则返回的张量上的操作会被 autograd 追踪，用于计算梯度,否则不会。默认值为 False。


返回
:::::::::

``paddle.Tensor``，一维张量，大小为 `(window_length,)`，表示 Hamming 窗口。

代码示例
:::::::::

COPY-FROM: paddle.hamming_window
