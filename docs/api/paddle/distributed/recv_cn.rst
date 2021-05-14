.. _cn_api_distributed_recv:

send
-------------------------------


.. py:function:: paddle.distributed.recv(tensor, src=0, group=None, use_calc_stream=True)

发送tensor到指定接收者。

参数
:::::::::
    - tensor (Tensor) - 接收数据的Tensor。数据类型为：float16、float32、float64、int32、int64。
    - src (int) - 发送者的标识符。
    - group (Group，可选) - new_group返回的Group实例，或者设置为None表示默认地全局组。默认值：None。
    - use_calc_stream (bool，可选) - 标识使用计算流还是通信流。默认值：True。

返回
:::::::::
无

代码示例
:::::::::
.. code-block:: python

        import paddle
        from paddle.distributed import init_parallel_env
        init_parallel_env()
        if paddle.distributed.ParallelEnv().rank == 0:
            data = paddle.to_tensor([7, 8, 9])
            paddle.distributed.send(data, dst=1)
        else:
            data = paddle.to_tensor([1, 2, 3])
            paddle.distributed.recv(data, src=0)
        out = data.numpy()
