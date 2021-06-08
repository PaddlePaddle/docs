.. _cn_api_distributed_get_rank:

get_rank
----------

..  py:function:: paddle.distributed.get_rank()

返回当前进程的rank。

当前进程rank的值等于环境变量 ``PADDLE_TRAINER_ID`` 的值，默认值为0。

返回
:::::::::
(int) 当前进程的rank。

代码示例
:::::::::
.. code-block:: python

    import paddle
    import paddle.distributed as dist

    # execute this command in terminal: export PADDLE_TRAINER_ID=0
    print("The rank is %d" % dist.get_rank())
    # The rank is 0
