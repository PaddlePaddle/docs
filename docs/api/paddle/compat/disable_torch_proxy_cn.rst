.. _cn_api_paddle_compat_disable_torch_proxy:

disable_torch_proxy
-------------------------------

.. py:function:: paddle.compat.disable_torch_proxy()

通过从 ``sys.meta_path`` 中移除 ``TorchProxyMetaFinder`` 来禁用 PyTorch 代理。
这可以防止 ``torch`` 导入被代理到 PaddlePaddle。

返回
:::::::::
    None

代码示例
:::::::::

COPY-FROM: paddle.compat.disable_torch_proxy
