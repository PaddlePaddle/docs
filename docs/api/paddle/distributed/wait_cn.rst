.. _cn_api_distributed_wait:

wait
-------------------------------


.. py:function:: wait(tensor, group=None, use_calc_stream=True)


同步通信组

参数
:::::::::
    - tensor (Tensor): 指定同步时的 tensor 对象
    - group (Group): 同步的通信组
    - use_calc_stream (bool): 同步计算流（True），或同步通信流（False），默认为 True

返回
:::::::::
None

代码示例
:::::::::
.. code-block:: python

        import paddle

        paddle.distributed.init_parallel_env()
        tindata = paddle.randn(shape=[2, 3])
        paddle.distributed.all_reduce(tindata, use_calc_stream=True)
        paddle.distributed.wait(tindata)

