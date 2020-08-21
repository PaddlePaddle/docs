.. _cn_api_distributed_all_reduce:

all_reduce
-------------------------------


.. py:function:: paddle.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=0, async_op=False)

进程组内所有进程的指定tensor进行归约操作，并返回给所以进程归约的结果。

参数
:::::::::
    - tensor (Tensor) - 归约操作的输入tensor，同时也会将归约结果返回至此tensor中。tensor的数据类型为 `float32` 、 `float64` 、 `int32` 或 `int64` 。
    - op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD，可选) - 归约的具体操作，比如求和，取最大值，取最小值和求乘积，默认为求和归约。
    - group (int，可选) - 工作的进程组编号，默认为0。
    - async_op (bool，可选) - 广播操作为同步或者异步，默认为同步。

返回
:::::::::
无

代码示例
:::::::::
.. code-block:: python

        import paddle
        from paddle.distributed import ReduceOp
        paddle.disable_static()
        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
        with fluid.dygraph.guard(place=place):
             paddle.distributed.init_process_group('nccl', 1000, 2, 1)
             if fluid.dygraph.ParallelEnv().local_rank == 0:
                 np_data = np.array([[4, 5, 6], [4, 5, 6]])
             else:
                 np_data = np.array([[1, 2, 3], [1, 2, 3]])
             data = paddle.to_tensor(np_data)
             paddle.distributed.all_reduce(data)
             out = data.numpy()
             # [[5, 7, 9], [5, 7, 9]]

