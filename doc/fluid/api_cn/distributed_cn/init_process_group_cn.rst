.. _cn_api_distributed_init_process_group:

init_process_group
-------------------------------


.. py:function:: paddle.distributed.init_process_group(backend, timeout, rank_num, rank, store=None, group_name='')

初始化分布式环境

参数
:::::::::
    - backend (str) - 当前要使用的通信后端，可选择'nccl' 或者'gloo'。
    - rank_num (int) - 分布式进程组中的总进程个数。
    - rank (int) - 当前进程的排序编号，从0开始计数。
    - timeout (int) - 每个进程的超时时间。
    - group_name (str，可选) - 进程组的名字，默认为空。

返回
:::::::::
无

代码示例
:::::::::
.. code-block:: python

        import paddle
        import paddle.fluid as fluid
        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
        with fluid.dygraph.guard(place=place):
            paddle.distributed.init_process_group('nccl', 100, 2, 1)
