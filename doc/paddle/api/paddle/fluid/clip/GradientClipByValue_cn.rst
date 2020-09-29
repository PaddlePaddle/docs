.. _cn_api_fluid_clip_GradientClipByValue:

GradientClipByValue
-------------------------------

.. py:class:: paddle.nn.GradientClipByValue(max, min=None, need_clip=None)




将输入的多维Tensor :math:`X` 的值限制在 [min, max] 范围。

输入的 Tensor 不是从该类里传入， 而是默认会选择 ``Program`` 中全部的梯度，如果 ``need_clip`` 不为None，则可以只选择部分参数进行梯度裁剪。

该类需要在初始化 ``optimizer`` 时进行设置后才能生效，可参看 ``optimizer`` 文档(例如： :ref:`cn_api_fluid_optimizer_SGD` )。

给定一个 Tensor  ``t`` ，该操作将它的值压缩到 ``min`` 和 ``max`` 之间

- 任何小于 ``min`` 的值都被设置为 ``min``

- 任何大于 ``max`` 的值都被设置为 ``max``

参数:
 - **max** (foat) - 要修剪的最大值。
 - **min** (float，optional) - 要修剪的最小值。如果用户没有设置，将被自动设置为 ``-max`` （此时 ``max`` 必须大于0）。
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
    clip = paddle.nn.GradientClipByValue(min=-1, max=1)

    # 仅裁剪参数linear_0.w_0时：
    # pass a function(fileter_func) to need_clip, and fileter_func receive a ParamBase, and return bool
    # def fileter_func(ParamBase):
    # # 可以通过ParamBase.name判断（name可以在paddle.ParamAttr中设置，默认为linear_0.w_0、linear_0.b_0）
    #   return ParamBase.name == "linear_0.w_0"
    # # 注：linear.weight、linear.bias能分别返回dygraph.Linear层的权重与偏差，可以此来判断
    #   return ParamBase.name == linear.weight.name
    # clip = paddle.nn.GradientClipByValue(min=-1, max=1, need_clip=fileter_func)

    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
    sdg.step()
            
