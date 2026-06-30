.. _cn_api_paddle_inference_mode:

inference_mode
-------------------------------

.. py:class:: paddle.inference_mode(mode=True)



创建一个上下文管理器或装饰器，用于启用或禁用推理模式。

在该模式下，每次计算的结果都将具有 ``stop_gradient=True``。当 ``mode=False`` 时，将启用梯度计算。

也可以用作一个装饰器。

参数
::::::::::::

    - **mode** (bool，可选) - 是否启用推理模式。默认值为 True。

代码示例
::::::::::::

COPY-FROM: paddle.inference_mode
