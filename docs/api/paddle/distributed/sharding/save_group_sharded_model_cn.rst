.. _cn_api_paddle_distributed_sharding_save_group_sharded_model:

save_group_sharded_model
-------------------------------


.. py:function:: paddle.distributed.sharding.save_group_sharded_model(model, output, optimizer=None)

使用 save_group_sharded_model 可以对 group_sharded_parallel 配置后的模型和优化器状态进行保存。

.. note::
    此处需要注意，使用 save_group_sharded_model 保存模型，再次 load 时需要在调用 group_sharded_parallel 前对 model 和 optimizer 进行 set_state_dict。


参数
:::::::::
    - **model** (Layer) - 使用 group_sharded_parallel 配置后的模型。
    - **output** (str) - 输出保存模型和优化器的文件夹路径。
    - **optimizer** (Optimizer，可选) - 使用 group_sharded_parallel 配置后的优化器，默认为 None，表示不对优化器状态进行保存。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.sharding.save_group_sharded_model
