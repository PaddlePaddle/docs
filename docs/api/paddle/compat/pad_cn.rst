.. _cn_api_paddle_compat_pad:

pad
-------------------------------

.. py:function:: paddle.compat.pad(input, pad, mode='constant', value=0.0)

PyTorch 兼容的 :ref:`cn_api_paddle_nn_functional_pad` 版本

使用前请详细参考：`【paddle 参数更多】torch.nn.functional.pad`_ 以确定是否使用此模块。

.. _【paddle 参数更多】torch.nn.functional.pad: https://www.paddlepaddle.org.cn/documentation/docs/en/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.pad.html

根据 ``'pad'`` 和 ``'mode'`` 对输入张量进行 padding 操作。所有操作都是从 **最右侧的维度** （最后一个维度）开始。

.. note::
    此 API 遵循 ``torch.nn.functional.pad`` 的函数签名和行为以实现 PyTorch 兼容。
    如需使用 Paddle 原生实现，请参考 :ref:`cn_api_paddle_nn_functional_pad`

参数
:::::::::
       - **input** (Tensor) - 输入 N 维 Tensor，支持 float32、float64、int32、int64、complex64、complex128
       - **pad** (Tensor|list[int]|tuple[int]) - 填充大小，基本数据类型是 ``int32``。
       - **mode** (str, 可选) - - padding 的四种模式，分别为 ``'constant'``、``'reflect'``、``'replicate'`` 和 ``'circular'``，默认值为 ``constant``

         - ``'constant'`` 表示填充常数 ``value``；
         - ``'reflect'`` 表示填充以 ``input`` 边界值为轴的映射；
         - ``'replicate'`` 表示填充 ``input`` 边界值；
         - ``'circular'`` 为循环填充 ``input``。

       - **value** (float, 可选) - 以 ``'constant'`` 模式填充区域时填充的值，使用其他模式时此值不被使用。默认值为 :math:`0.0`。

.. note::

    对于非 ``'constant'``, ``pad`` 的尺寸不能超过 ``min(2 * input.ndim - 2, 6)``。
    此外非 ``'constant'`` 中只有 2D、3D、4D 以及 5D 张量受到支持，且至多仅能操作最后三个维度（当 ``ndim>=3`` 时）。

返回
:::::::::
Tensor，对 ``input`` 进行 ``'pad'`` 的结果，数据类型和 ``input`` 相同。


代码示例
:::::::::

COPY-FROM: paddle.compat.pad
