.. _cn_api_paddle_compat_extend_torch_proxy_blocked_modules:

extend_torch_proxy_blocked_modules
-------------------------------

.. py:function:: paddle.compat.extend_torch_proxy_blocked_modules(modules)

将模块添加到 PyTorch 代理禁用列表中。

禁用列表中的模块在导入时不会使用 PyTorch 代理，并且它们的函数在调用时也不会触发 PyTorch 代理。

默认情况下，一些模块已经在禁用列表中，例如 ``tvm_ffi``。

参数
:::::::::
    - **modules** (Iterable[str]) - 要从 PyTorch 代理中阻止的模块名称的可迭代对象。

返回
:::::::::
    None

代码示例
:::::::::

COPY-FROM: paddle.compat.extend_torch_proxy_blocked_modules
