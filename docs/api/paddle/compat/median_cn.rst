.. _cn_api_paddle_compat_median:

median
-------------------------------

.. py:function:: paddle.compat.median(input, dim=None, keepdim=False, *, out=None)

PyTorch 兼容的 :ref:`cn_api_paddle_median` 版本，提供完全一致的函数签名与行为：
- 当  ``dim``  为  ``None``  时，返回所有元素的中位数。
- 当  ``dim``  指定维度时，返回该维度的中位数与对应索引。

.. note::
   此 API 遵循  ``torch.median``  的函数签名和行为以实现 PyTorch 兼容。如需使用 Paddle 原生实现，请参考 :ref:`cn_api_paddle_median`。

参数
::::::::::
- **input** (Tensor) - 输入 N 维 Tensor，支持 bool、bfloat16、float16、float32、float64、int32、int64 数据类型。
- **dim** (int，可选) - 指定计算中位数的维度。为  ``None``  时计算全局中位数。默认  ``None`` 。
- **keepdim** (bool，可选) - 是否保留被约简的维度。默认  ``False`` 。
- **out** (tuple(Tensor, Tensor)|Tensor，可选) - 关键字参数。当指定  ``dim``  时，可传入二元组  ``(values, indices)``  用于原位写回中位数与索引；当未指定  ``dim``  时，可传入单个  ``Tensor``  用于写回标量结果。默认  ``None`` 。

返回
::::::::::
- 当  ``dim``  为  ``None`` ：返回一个标量  ``Tensor`` ，为  ``input``  的中位数。
- 当  ``dim``  指定：返回具名元组  ``(values, indices)`` ，分别是中位数与其索引。

代码示例
::::::::::

COPY-FROM: paddle.compat.median
