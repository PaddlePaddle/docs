.. _cn_api_paddle_incubate_distributed_utils_io_dist_save_save_for_auto_inference:

save_for_auto_inference
-------------------------------

.. py:function:: paddle.incubate.distributed.utils.io.dist_save.save_for_auto_inference(path_prefix, dist_model, cvt2cpu=False)

描述
:::::::::
    保存模型参数以进行自动并行推理。支持 dp + mp + pp + sharding(stage1)、dp + sharding stage2-3。
    在自动并行模式下支持 MoE 之前, MoE 不会被支持。


参数
:::::::::
    - **path_prefix**:  要保存的路径前缀。如果 `path_preifx` 以路径分隔符结尾, 则路径将作为目录进行处理，参数将保存在其中，并自动命名为 saved_parameters。除此之外，参数将保存为名称 path_preifx_dist{global_rank}.pdparams 和 path_preifx_dist{global_rank}.pdattrs。
    - **dist_model**: 分布式模型中的模型。
    - **cvt2cpu**: 在使用 sharding stage3 时将参数移动到 CPU。如果不使用 sharding stage3, 则 var 无效。


返回
:::::::::
    None


代码示例
::::::::::
    COPY-FROM: paddle.incubate.distributed.utils.io.dist_save.save_for_auto_inference


输出
:::::::::
    path/to/save_infer_dist0.pdparams path/to/save_infer_dist1.pdparams path/to/save_infer_dist2.pdparams ...
    path/to/save_infer_dist0.pdattr  path/to/save_infer_dist1.pdattr   path/to/save_infer_dist2.pdattr ...
