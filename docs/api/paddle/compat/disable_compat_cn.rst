.. _cn_api_paddle_disable_compat:

disable_compat
-------------------------------

.. py:function:: paddle.disable_compat()

通过从 ``sys.meta_path`` 中移除 ``TorchProxyMetaFinder`` 来禁用 PyTorch 代理。
这可以防止 ``torch`` 导入被代理到 PaddlePaddle。

返回
:::::::::
    None

代码示例
:::::::::

COPY-FROM: paddle.disable_compat
