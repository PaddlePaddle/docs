.. _cn_api_paddle_flops:

flops
-------------------------------

.. py:function:: paddle.flops(net, input_size, custom_ops=None, print_detail=False)

打印网络的基础结构和参数信息。

参数
:::::::::
  - **net** (paddle.nn.Layer|paddle.static.Program) - 网络实例，必须是 paddle.nn.Layer 的子类或者静态图下的 paddle.static.Program。
  - **input_size** (list) - 输入 Tensor 的大小。注意：仅支持 batch_size=1。
  - **custom_ops** (dict，可选) - 字典，用于实现对自定义网络层的统计。字典的 key 为自定义网络层的 class，value 为统计网络层 flops 的函数，函数实现方法见示例代码。此参数仅在 ``net`` 为 paddle.nn.Layer 时生效。默认值：None。
  - **print_detail** (bool，可选) - bool 值，用于控制是否打印每个网络层的细节。默认值：False。

返回
:::::::::
int，网络模型的计算量。

代码示例
:::::::::

COPY-FROM: paddle.flops
