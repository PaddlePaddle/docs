.. _cn_api_fluid_clip_ClipGradByValue:

ClipGradByValue
-------------------------------

.. py:class:: paddle.nn.ClipGradByValue(max, min=None)




将输入的多维 Tensor :math:`X` 的值限制在 [min, max] 范围。

输入的 Tensor 不是从该类里传入，而是默认选择优化器中输入的所有参数的梯度。如果某个参数 ``ParamAttr`` 中的 ``need_clip`` 值被设置为 ``False``，则该参数的梯度不会被裁剪。

该类需要在初始化 ``optimizer`` 时进行设置后才能生效，可参看 ``optimizer`` 文档(例如：:ref:`cn_api_paddle_optimizer_SGD` )。

给定一个 Tensor  ``t``，该操作将它的值压缩到 ``min`` 和 ``max`` 之间

- 任何小于 ``min`` 的值都被设置为 ``min``

- 任何大于 ``max`` 的值都被设置为 ``max``

.. note::
   ``ClipGradByValue`` 的 ``need_clip`` 方法从 2.0 开始废弃。请在 :ref:`paddle.ParamAttr <cn_api_fluid_ParamAttr>` 中使用 ``need_clip`` 来说明 ``clip`` 范围。

参数
::::::::::::

 - **max** (float) - 要修剪的最大值。
 - **min** (float，可选) - 要修剪的最小值。如果用户没有设置，将被自动设置为 ``-max`` （此时 ``max`` 必须大于 :math:`0`）。

代码示例
::::::::::::

COPY-FROM: paddle.nn.ClipGradByValue
