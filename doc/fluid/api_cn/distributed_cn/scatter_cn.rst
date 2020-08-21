.. _cn_api_distributed_scatter:

scatter
-------------------------------


.. py:function:: paddle.distributed.scatter(tensor, tensor_list=None, src=0, group=0, async_op=False)

进程组内指定进程源的tensor列表分发到其他所有进程中。

参数
:::::::::
    - tensor (Tensor) - 分发操作的输出tensor。tensor的数据类型为 `float32` 、 `float64` 、 `int32` 或 `int64` 。
    - tensor_list (list，可选) - 分发操作的输入tensor列表，默认为None。列表中的每个元素均为tensor，每个tensor的数据类型为 `float32` 、 `float64` 、 `int32` 或 `int64` 。
    - src (int，可选) - 分发操作的源进程号，该进程号的tensor列表将分发到其他进程中。默认为0。
    - group (int，可选) - 工作的进程组编号，默认为0。
    - async_op (bool，可选) - 广播操作为同步或者异步，默认为同步。

返回
:::::::::
无

代码示例
:::::::::
.. code-block:: python

        import paddle
        paddle.disable_static()
        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
        with fluid.dygraph.guard(place=place):
             paddle.distributed.init_process_group('nccl', 1000, 2, 1)
             if fluid.dygraph.ParallelEnv().local_rank == 0:
                 np_data1 = np.array([7, 8, 9])
                 np_data2 = np.array([10, 11, 12])
             else:
                 np_data1 = np.array([1, 2, 3])
                 np_data2 = np.array([4, 5, 6])
             data1 = paddle.to_tensor(np_data1)
             data2 = paddle.to_tensor(np_data2)
             if fluid.dygraph.ParallelEnv().local_rank == 0:
                 paddle.distributed.scatter(data1, src=1)
             else:
                 paddle.distributed.scatter(data1, tensor_list=[data1, data2], src=1)
             out = data1.numpy()

