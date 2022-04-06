.. _cn_api_distributed_sharding_save_group_sharded_model:

save_group_sharded_model
-------------------------------


.. py:function:: paddle.distributed.sharding.save_group_sharded_model(model, output, optimizer=None)

使用save_group_sharded_model可以对group_sharded_parallel配置后的模型和优化器状态进行保存。

.. note::
    此处需要注意，使用save_group_sharded_model保存模型，再次load时需要在调用group_sharded_parallel前对model和optimizer进行set_state_dict。


参数
:::::::::
    - model (Layer) - 使用group_sharded_parallel配置后的模型。
    - output (str) - 输出保存模型和优化器的文件夹路径。
    - optimizer (Optimizer，可选) - 使用group_sharded_parallel配置后的优化器，默认为None，表示不对优化器状态进行保存。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.sharding.save_group_sharded_model
