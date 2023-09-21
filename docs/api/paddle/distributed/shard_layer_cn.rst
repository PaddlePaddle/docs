.. _cn_api_paddle_distributed_shard_layer:

shard_layer
-------------------------------

.. py:function:: paddle.distributed.shard_layer(layer, process_mesh, shard_fn=None, input_fn=None, output_fn=None)

根据参数 `shard_fn` 将传入的 `paddle.nn.Layer` 所有的参数转换为带有分布式切分信息的 `Tensor`。同时也支持指定 `input_fn` 和 `output_fn` 用于控制输入和输出 `Tensor` 的转换。(具体指的是，将输入转换为带有分布式切分信息的 `Tensor`，将输出转回不带分布式切分信息的 `Tensor`。)

`shard_fn` 的函数签名为：def shard_fn(layer_name, layer, process_mesh) -> None。
`input_fn` 的函数签名为：def input_fn(inputs, process_mesh) -> list(paddle.Tensor)，一般地，`input_fn` 返回值的类型为带有分布式切分信息的 `Tensor`。
`output_fn` 的函数签名为：def output_fn(outputs, process_mesh) -> list(paddle.Tensor)，一般地，`output_fn` 返回值的类型为不带分布式切分信息的 `Tensor`。


参数
:::::::::

    - **layer** (paddle.nn.Layer) - 需要被切分的 `Layer` 对象。
    - **process_mesh** (paddle.distributed.ProcessMesh) - 执行当前 `Layer` 的 `ProcessMesh` 信息。
    - **shard_fn** (Callable) - 用于切分当前 `Layer` 参数的函数。如果没有指定，默认地我们将在当前 `ProcessMesh` 上复制所有的参数。
    - **input_fn** (Callable) - 指定如何切分 `Layer` 的输入。`input_fn` 函数将被注册为 `Layer` 的一个 `forward pre-hook`。默认我们将不会切分 `Layer` 的输入。
    - **output_fn** (Callable) - 指定如何切分 `Layer` 的输出，或者将 `Layer` 的输出转回不带分布式切分信息的 `Tensor`。`output_fn` 函数将被注册为 `Layer` 的一个 `forward post-hook`。默认我们将不会切分或者转换 `Layer` 的输出。

返回
:::::::::
Layer：一个参数全部为带有分布式切分信息 `Tensor` 的 `Layer` 对象。


代码示例
:::::::::

COPY-FROM: paddle.distributed.shard_layer
