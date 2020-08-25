.. _cn_api_distributed_barrier:

barrier
-------------------------------


.. py:function:: paddle.distributed.barrier(group=0)

同步进程组内的所有进程。

参数
:::::::::
    - group (int，可选) - 工作的进程组编号，默认为0。

返回
:::::::::
无

代码示例
:::::::::
.. code-block:: python

        import paddle
        import paddle.fluid as fluid
        from fluid.dygraph.parallel import prepare_context


        paddle.disable_static()
        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
        with fluid.dygraph.guard(place=place):
            prepare_context()
            paddle.distributed.barrier()
