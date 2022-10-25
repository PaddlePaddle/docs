.. _cn_api_distributed_fleet_utils_recompute:

recompute
-------------------------------


.. py:function:: paddle.distributed.fleet.utils.recompute(function, *args, **kwargs)

重新计算中间激活函数值来节省显存。

参数
:::::::::
    - **function** (paddle.nn.Sequential) - 模型前向传播的部分连续的层函数组成的序列，它们的中间激活函数值将在前向传播过程中被释放掉来节省显存，并且在反向梯度计算的时候会重新被计算。
    - **args** (Tensor) - function 的输入。
    - **kwargs** (Dict) - kwargs 只应该包含 preserve_rng_state 的键值对，用来表示是否保存前向的 rng，如果为 True，那么在反向传播的重计算前向时会还原上次前向的 rng 值。默认 preserve_rng_state 为 True。

返回
:::::::::
function 作用在输入的输出

代码示例
:::::::::
COPY-FROM: paddle.distributed.fleet.utils.recompute
