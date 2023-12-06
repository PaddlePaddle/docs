.. _cn_api_paddle_distributed_fleet_utils_recompute:

recompute
-------------------------------


.. py:function:: paddle.distributed.fleet.utils.recompute(function, *args, **kwargs)

重新计算中间激活函数值来节省显存。

参数
:::::::::
    - **function** (paddle.nn.Layer) - 模型前向传播的部分连续的层函数组成的序列，它们的中间激活函数值将在前向传播过程中被释放掉来节省显存，并且在反向梯度计算的时候会重新被计算。
    - **args** (Tensor) - function 的输入。
    - **kwargs** (Dict) - kwargs 只应该包含两类键值对。一类键值是 function 的字典参数，另外一类仅只能包含 preserve_rng_state 和 use_reentrant 两个 key。 preserve_rng_state 的键值对，用来表示是否保存前向的 rng，如果为 True，那么在反向传播的重计算前向时会还原上次前向的 rng 值。默认 preserve_rng_state 为 True。 use_reentrant 的键值对，用来表示 recompute 的实现方式，如果为 True，意味着 recompute 使用 PyLayer 的方式实现的，如果为 False， recompute 内部则使用 hook 的方式实现的，默认值是 True。在某些场景下，比如 recompute 与数据并行结合时，需要额外调用 no_sync 函数，此时可以设置 use_reentrant=False，选用 hook 方式的 recompute，可以避免额外调用 no_sync 函数。

返回
:::::::::
function 作用在输入的输出

代码示例
:::::::::
COPY-FROM: paddle.distributed.fleet.utils.recompute
