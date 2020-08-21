.. _cn_api_distributed_barrier:

barrier
-------------------------------


.. py:function:: paddle.distributed.barrier(group=0, async_op=False)

同步进程组内的所有进程。

参数
:::::::::
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
             paddle.distributed.barrier()
