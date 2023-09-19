.. _cn_api_distributed_shard_layer:

shard_layer
-------------------------------

.. py:class:: paddle.distributed.shard_layer(model, process_mesh,shard_fn:, input_fn, output_fn)

将一个paddle.nn.layer创建的model按照shard_fn和process_mesh改为参数均为DistTensor的 model

参数
:::::::::

    - **model**  - 用nn.Layer组建的model
    - **process_mesh**  - 要放置的进程组信息
    - **shard_fn**  - 切分model参数的函数。如果未指定，默认情况下跨ProcessMesh，复制model的所有参数。
    - **input_fn**  - 指定输入的切分分布，input_fn将作为Layer的forward_pre_hook 
    - **output_fn**  - 指定输出的切分分布，output_fn将作为Layer的forward_post_hook 

返回
:::::::::
参数均为DistTensor的 model



**代码示例**

COPY-FROM: paddle.distributed.dtensor_from_fn
