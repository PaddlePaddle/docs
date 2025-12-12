.. _cn_api_paddle_compat_use_torch_proxy_guard:

use_torch_proxy_guard
-------------------------------

.. py:function:: paddle.compat.use_torch_proxy_guard(*, enable=True, scope=None, silent=False)

用于临时启用或禁用 PyTorch 代理的上下文管理器。

当 ``enable`` 为 ``True``（默认值）时，PyTorch 代理在上下文持续时间内启用，并在之后恢复到之前的状态。

当 ``enable`` 为 ``False`` 时，PyTorch 代理在上下文持续时间内禁用，并在之后恢复。

参数
:::::::::
    - **enable** (bool，可选) - 是否在上下文中启用或禁用 PyTorch 代理。默认为 ``True``。
    - **scope** (str|Iterable[str]，可选) - 指定启用 PyTorch 代理的模块或模块列表。如果为 ``None``，则使用全局范围。默认为 ``None``。
    - **silent** (bool，可选) - 如果为 True，则抑制有关范围更改的警告。默认为 ``False``。

代码示例
:::::::::

COPY-FROM: paddle.compat.use_torch_proxy_guard
