.. _cn_api_fluid_layers_Print:

Print
-------------------------------


.. py:function:: paddle.static.Print(input, first_n=-1, message=None, summarize=20, print_tensor_name=True, print_tensor_type=True, print_tensor_shape=True, print_tensor_lod=True, print_phase='both')




创建一个打印操作，打印正在访问的 Tensor 内容。

封装传入的 Tensor，以便无论何时访问 Tensor，都会打印信息 message 和 Tensor 的当前值。

参数
::::::::::::

    - **input** (Variable)-将要打印的 Tensor。
    - **summarize** (int)-打印 Tensor 中的元素数目，如果值为-1 则打印所有元素。默认值为 20。
    - **message** (str)-打印 Tensor 信息前自定义的字符串类型消息，作为前缀打印。
    - **first_n** (int)-打印 Tensor 的次数。
    - **print_tensor_name** (bool，可选)-指明是否打印 Tensor 名称，默认为 True。
    - **print_tensor_type** (bool，可选)-指明是否打印 Tensor 类型，默认为 True。
    - **print_tensor_shape** (bool，可选)-指明是否打印 Tensor 维度信息，默认为 True。
    - **print_tensor_lod** (bool，可选)-指明是否打印 Tensor 的 LoD 信息，默认为 True。
    - **print_phase** (str，可选)-指明打印的阶段，包括 ``forward`` , ``backward`` 和 ``both``，默认为 ``both``。设置为 ``forward`` 时，只打印 Tensor 的前向信息；设置为 ``backward`` 时，只打印 Tensor 的梯度信息；设置为 ``both`` 时，则同时打印 Tensor 的前向信息以及梯度信息。

返回
::::::::::::
输出 Tensor。

.. note::
   输入和输出是两个不同的 Variable，在接下来的过程中，应该使用输出 Variable 而非输入 Variable，否则打印层将失去 backward 的信息。

代码示例
::::::::::::

COPY-FROM: paddle.static.Print
