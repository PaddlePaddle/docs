.. _cn_api_paddle_compat_slogdet:

slogdet
-------------------------------

.. py:function:: paddle.compat.slogdet(x, out=None)

PyTorch 兼容的 :ref:`cn_api_paddle_slogdet` 版本，计算方阵或批量方阵行列式的符号与绝对值的自然对数，行列式的值可以通过 ``sign * exp(logabsdet)`` 复原。

使用前请详细参考：`【返回参数类型不一致】torch.slogdet`_ 以确定是否使用此模块。

.. _【返回参数类型不一致】torch.slogdet: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/torch/torch.slogdet.html


参数
::::::::::::

    - **x** (Tensor) - 输入的多维 ``Tensor``，支持的数据类型：float32, float64, complex64, complex128。
    - **out** (tuple(Tensor, Tensor)，可选) - 用于引用式传入输出值。``sign`` 在前，``logabsdet`` 在后。默认值为 None。``out`` 返回方法与静态图联合使用是被禁止的行为，静态图下将报错。

返回
::::::::::::

SlogdetResult(Tensor, Tensor)，此处的 ``SlogdetResult`` 是一个具名元组，含有 ``sign`` （在前）和 ``logabsdet`` （在后）两个域，用法与 tuple 一致。


代码示例
::::::::::::

COPY-FROM: paddle.compat.slogdet
