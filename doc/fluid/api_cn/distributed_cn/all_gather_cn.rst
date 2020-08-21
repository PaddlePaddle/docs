.. _cn_api_distributed_all_gather:

all_gather
-------------------------------


.. py:function:: paddle.distributed.all_gather(tensor_list, tensor, group=0, async_op=False)

进程组内所有进程的指定tensor进行聚合操作，并返回给所以进程聚合的结果。

参数
:::::::::
    - tensor_list (list) - 聚合操作的输出tensor列表。列表中的每个元素均为tensor，每个tensor的数据类型为 `float32` 、 `float64` 、 `int32` 或 `int64` 。
    - tensor (Tensor) - 聚合操作的输入tensor。tensor的数据类型为 `float32` 、 `float64` 、 `int32` 或 `int64` 。
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
             tensor_list = []
             if fluid.dygraph.ParallelEnv().local_rank == 0:
                 np_data1 = np.array([[4, 5, 6], [4, 5, 6]])
                 np_data2 = np.array([[4, 5, 6], [4, 5, 6]])
                 data1 = paddle.to_tensor(np_data1)
                 data2 = paddle.to_tensor(np_data2)
                 paddle.distributed.all_gather(tensor_list, data1)
             else:
                 np_data1 = np.array([[1, 2, 3], [1, 2, 3]])
                 np_data2 = np.array([[1, 2, 3], [1, 2, 3]])
                 data1 = paddle.to_tensor(np_data1)
                 data2 = paddle.to_tensor(np_data2)
                 out = paddle.distributed.all_gather(tensor_list, data2)
