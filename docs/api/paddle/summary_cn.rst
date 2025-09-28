.. _cn_api_paddle_summary:

summary
-------------------------------

.. py:function:: paddle.summary(net, input_size=None, dtypes=None, input=None)

通过 ``input_size`` 或 ``input`` 打印网络 ``net`` 的基础结构和参数信息。``input_size`` 指定网络 ``net`` 输入 Tensor 的大小，而 ``input`` 指定网络 ``net`` 的输入 Tensor；如果给出 ``input``，那么 ``input_size`` 和 ``dtypes`` 的输入将被忽略。


参数
:::::::::
  - **net** (Layer) - 网络实例，必须是 ``Layer`` 的子类。
  - **input_size** (tuple|InputSpec|list[tuple|InputSpec，可选) - 输入 Tensor 的大小。如果网络只有一个输入，那么该值需要设定为 tuple 或 InputSpec。如果模型有多个输入。那么该值需要设定为 list[tuple|InputSpec]，包含每个输入的 shape。默认值：None。
  - **dtypes** (str，可选) - 输入 Tensor 的数据类型，如果没有给定，默认使用 ``float32`` 类型。默认值：None。
  - **input** (tensor，可选) - 输入的 Tensor，如果给出 ``input``，那么 ``input_size`` 和 ``dtypes`` 的输入将被忽略。默认值：None。

返回
:::::::::
字典，包含了总的参数量和总的可训练的参数量。

代码示例 1
:::::::::

COPY-FROM: paddle.summary:code-example-1

代码示例 2
:::::::::

COPY-FROM: paddle.summary:code-example-2

代码示例 3
:::::::::

COPY-FROM: paddle.summary:code-example-3
