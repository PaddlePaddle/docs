.. _cn_api_fluid_clip_ClipGradByNorm:

ClipGradByNorm
-------------------------------

.. py:class:: paddle.nn.ClipGradByNorm(clip_norm)




将输入的多维 Tensor :math:`X` 的 L2 范数限制在 ``clip_norm`` 范围之内。

- 如果 L2 范数大于 ``clip_norm``，则该 Tensor 会乘以一个系数进行压缩

- 如果 L2 范数小于或等于 ``clip_norm``，则不会进行任何操作。

输入的 Tensor 不是从该类里传入，而是默认选择优化器中输入的所有参数的梯度。如果某个参数 ``ParamAttr`` 中的 ``need_clip`` 值被设置为 ``False``，则该参数的梯度不会被裁剪。

该类需要在初始化 ``optimizer`` 时进行设置后才能生效，可参看 ``optimizer`` 文档(例如：:ref:`cn_api_paddle_optimizer_SGD` )。

裁剪公式如下：

.. math::

  Out=
  \left\{
  \begin{aligned}
  &  X & & if (norm(X) \leq clip\_norm)\\
  &  \frac{clip\_norm∗X}{norm(X)} & & if (norm(X) > clip\_norm) \\
  \end{aligned}
  \right.


其中 :math:`norm（X）` 代表 :math:`X` 的 L2 范数

.. math::
  \\norm(X) = (\sum_{i=1}^{n}|x_i|^2)^{\frac{1}{2}}\\

.. note::
   ``ClipGradByNorm`` 的 ``need_clip`` 方法从 2.0 开始废弃。请在 :ref:`paddle.ParamAttr <cn_api_fluid_ParamAttr>` 中使用 ``need_clip`` 来说明 ``clip`` 范围。

参数
::::::::::::

 - **clip_norm** (float) - 所允许的二范数最大值。

代码示例
::::::::::::

COPY-FROM: paddle.nn.ClipGradByNorm
