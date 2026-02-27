.. _cn_api_paddle_enable_compat:

enable_compat
-------------------------------

.. py:function:: paddle.enable_compat(*, scope=None, blocked_modules=None, backend='torch', silent=False)

通过将 ``TorchProxyMetaFinder`` 添加到 ``sys.meta_path`` 来启用 PyTorch 兼容代理。
这允许导入实际上是 PaddlePaddle 代理的 ``torch`` 模块。


参数
:::::::::
    - **scope** (str|Iterable[str]，可选) - 指定启用 PyTorch 兼容代理的模块或模块列表。如果为 ``None``，则全局启用 PyTorch 兼容代理。默认为 ``None``。
    - **blocked_modules** (str|Iterable[str]，可选) - 指定从 PyTorch 兼容代理中排除的模块或模块列表。默认为 ``None``。
    - **backend** (str，可选) - 要启用兼容性的后端。目前仅支持 ``"torch"``。默认为 ``"torch"``。
    - **silent** (bool，可选) - 如果为 ``True``，则抑制有关范围更改的警告。默认为 ``False``。

返回
:::::::::
    None

代码示例 1
:::::::::

COPY-FROM: paddle.enable_compat:enable-compat-in-global-scope

代码示例 2
:::::::::

COPY-FROM: paddle.enable_compat:enable-compat-in-specific-scope
