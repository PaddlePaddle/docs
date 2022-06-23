.. _cn_api_paddle_tensor_maximum:

maximum
-------------------------------

.. py:function:: paddle.maximum(x, y, name=None)


逐元素对比输入的两个Tensor，并且把各个位置更大的元素保存到返回结果中。

等式是：

.. math::
        out = max(x, y)

.. note::
   ``paddle.maximum`` 遵守broadcasting，如您想了解更多，请参见 :ref:`cn_user_guide_broadcasting` 。

参数
:::::::::
   - **x** （Tensor）- 输入的Tensor。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
   - **y** （Tensor）- 输入的Tensor。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
   ``Tensor``，存储运算后的结果。如果x和y有不同的shape且是可以广播的，返回Tensor的shape是x和y经过广播后的shape。如果x和y有相同的shape，返回Tensor的shape与x，y相同。


代码示例
::::::::::

COPY-FROM: paddle.maximum