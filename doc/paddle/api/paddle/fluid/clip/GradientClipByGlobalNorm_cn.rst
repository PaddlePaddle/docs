.. _cn_api_fluid_clip_GradientClipByGlobalNorm:

GradientClipByGlobalNorm
-------------------------------

.. py:class:: paddle.nn.GradientClipByGlobalNorm(clip_norm, group_name='default_group', need_clip=None)



 
将一个 Tensor列表 :math:`t\_list` 中所有Tensor的L2范数之和，限定在 ``clip_norm`` 范围内。

- 如果范数之和大于 ``clip_norm`` ，则所有 Tensor 会乘以一个系数进行压缩

- 如果范数之和小于或等于 ``clip_norm`` ，则不会进行任何操作。

输入的 Tensor列表 不是从该类里传入， 而是默认会选择 ``Program`` 中全部的梯度，如果 ``need_clip`` 不为None，则可以只选择部分参数进行梯度裁剪。

该类需要在初始化 ``optimizer`` 时进行设置后才能生效，可参看 ``optimizer`` 文档(例如： :ref:`cn_api_fluid_optimizer_SGD` )。

裁剪公式如下：

.. math::
            \\t\_list[i]=t\_list[i]∗\frac{clip\_norm}{max(global\_norm,clip\_norm)}\\
            
其中：

.. math::            
            \\global\_norm=\sqrt{\sum_{i=0}^{n-1}(l2norm(t\_list[i]))^2}\\


参数:
 - **clip_norm** (float) - 所允许的范数最大值
 - **group_name** (str, optional) - 剪切的组名
 - **need_clip** (function, optional) - 类型: 函数。用于指定需要梯度裁剪的参数，该函数接收一个 ``Parameter`` ，返回一个 ``bool`` (True表示需要裁剪，False不需要裁剪)。默认为None，此时会裁剪网络中全部参数。

**代码示例**
 
.. code-block:: python

    import paddle
    import numpy as np

    # used in default dygraph mode

    paddle.disable_static()

    x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
    linear = paddle.nn.Linear(10, 10)
    out = linear(x)
    loss = paddle.mean(out)
    loss.backward()

    # 裁剪网络中全部参数：
    clip = paddle.nn.GradientClipByGlobalNorm(clip_norm=1.0)

    # 仅裁剪参数linear_0.w_0时：
    # pass a function(fileter_func) to need_clip, and fileter_func receive a ParamBase, and return bool
    # def fileter_func(ParamBase):
    # # 可以通过ParamBase.name判断（name可以在paddle.ParamAttr中设置，默认为linear_0.w_0、linear_0.b_0）
    #   return ParamBase.name == "linear_0.w_0"
    # # 注：linear.weight、linear.bias能分别返回dygraph.Linear层的权重与偏差，可以此来判断
    #   return ParamBase.name == linear.weight.name
    # clip = paddle.nn.GradientClipByGlobalNorm(clip_norm=1.0, need_clip=fileter_func)

    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
    sdg.minimize(loss)
            
