.. _cn_api_paddle_compat_enable_torch_proxy:

enable_torch_proxy
-------------------------------

.. py:function:: paddle.compat.enable_torch_proxy(*, scope=None, silent=False)

通过将 ``TorchProxyMetaFinder`` 添加到 ``sys.meta_path`` 来启用 PyTorch 代理。
这允许导入实际上是 PaddlePaddle 代理的 ``torch`` 模块。

参数
:::::::::
    - **scope** (str|Iterable[str]，可选) - 指定启用 PyTorch 代理的模块或模块列表。如果为 ``None``，则全局启用 PyTorch 代理。默认为 ``None``。
    - **silent** (bool，可选) - 如果为 True，则抑制有关范围更改的警告。默认为 ``False``。

返回
:::::::::
    None

代码示例 1
:::::::::

COPY-FROM: paddle.compat.enable_torch_proxy:enable-torch-proxy-in-global-scope

代码示例 2
:::::::::

COPY-FROM: paddle.compat.enable_torch_proxy:enable-torch-proxy-in-specific-scope
