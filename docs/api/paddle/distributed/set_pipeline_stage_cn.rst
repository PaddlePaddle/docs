.. _cn_api_distributed_set_pipeline_stage:

shard_op
-------------------------------


.. py:function:: paddle.distributed.set_pipeline_stage(stage)

设置后续操作算子所属的流水线级数。

参数
:::::::::
    - stage (int) - 流水线级。

返回
:::::::::
空

代码示例
:::::::::
.. code-block:: python

    import paddle
    import paddle.distributed as dist

    paddle.enable_static()

    dist.set_pipeline_stage(1)
