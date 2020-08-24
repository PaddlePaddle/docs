.. _cn_api_distributed_init_distributed_context:

init_distributed_context
-------------------------------


.. py:function:: paddle.distributed.init_distributed_context(backend, rank_num, rank, timeout=999999, group_name='', group_num=1, fs_path="", fs_name="", fs_ugi="")

初始化分布式环境

参数
:::::::::
    - backend (str) - 当前要使用的通信后端，可选择'nccl' 或者'gloo'。
    - rank_num (int) - 分布式进程组中的总进程个数。
    - rank (int) - 当前进程的排序编号，从0开始计数。
    - timeout (int，可选) - 使用gloo后端的情况下每个进程的超时时间，单位为秒，默认值为999999。
    - group_name (str，可选) - 进程组的名字，默认为空。
    - group_num (int，可选) - 进程组的数量，默认为1。
    - fs_path (str，可选) - 初始化gloo时指定的文件系统路径，默认为空。
    - fs_name (str，可选) - 初始化gloo时指定的文件系统名字，默认为空。
    - fs_ugi (str，可选) - 初始化gloo时指定的文件系统ugi（名字和密码），默认为空。


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
            paddle.distributed.init_distributed_context('nccl', 100, 2, 1)
