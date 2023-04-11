.. _cn_api_fluid_clip_ClipGradByGlobalNorm:

ClipGradByGlobalNorm
-------------------------------

.. py:class:: paddle.nn.ClipGradByGlobalNorm(clip_norm, group_name='default_group', auto_skip_clip=False)




将一个 Tensor 列表 :math:`t\_list` 中所有 Tensor 的 L2 范数之和，限定在 ``clip_norm`` 范围内。

- 如果范数之和大于 ``clip_norm``，则所有 Tensor 会乘以一个系数进行压缩

- 如果范数之和小于或等于 ``clip_norm``，则不会进行任何操作。

输入的 Tensor 不是从该类里传入，而是默认选择优化器中输入的所有参数的梯度。如果某个参数 ``ParamAttr`` 中的 ``need_clip`` 值被设置为 ``False``，则该参数的梯度不会被裁剪。

该类需要在初始化 ``optimizer`` 时进行设置后才能生效，可参看 ``optimizer`` 文档(例如：:ref:`cn_api_paddle_optimizer_SGD` )。

裁剪公式如下：

.. math::
            \\t\_list[i]=t\_list[i]∗\frac{clip\_norm}{max(global\_norm,clip\_norm)}\\

其中：

.. math::
            \\global\_norm=\sqrt{\sum_{i=0}^{n-1}(l2norm(t\_list[i]))^2}\\

.. note::
   ``ClipGradByGlobalNorm`` 的 ``need_clip`` 方法从 2.0 开始废弃。请在 :ref:`paddle.ParamAttr <cn_api_fluid_ParamAttr>` 中使用 ``need_clip`` 来说明 ``clip`` 范围。

参数
::::::::::::

 - **clip_norm** (float) - 所允许的范数最大值
 - **group_name** (str, optional) - 剪切的组名
 - **auto_skip_clip** (bool, optional): 跳过剪切梯度。默认值为 False。

代码示例
::::::::::::

COPY-FROM: paddle.nn.ClipGradByGlobalNorm
