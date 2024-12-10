.. _cn_api_paddle_distributed_parallelize:

parallelize
-------------------------------

.. py:function:: paddle.distributed.parallelize(model, optimizer=None, mesh=None, config=None)

基于用户提供的 ``mesh`` 将单卡视角的模型与优化器（如提供）进行并行化处理，转为分布式视角的模型与优化器（如提供）。


.. note::
    如果提供的 ``config`` 的键中不包含 ``dp_config``，``mp_config`` 与 ``pp_config`` 中的任何字段，则本函数会直接返回 model 与 optimizer。


参数
:::::::::

    - **model** (paddle.nn.Layer) - 单卡视角的模型。
    - **optimizer** (paddle.optimizer.Optimizer，可选) - 单卡视角的优化器。默认值为 None。如不提供优化器，则仅对模型进行并行化处理。
    - **mesh** (paddle.distributed.ProcessMesh，可选) - 模型与优化器（如提供）并行所发生的进程空间。最佳用法：在调用本 API 之前，通过
      调用 ``paddle.distributed.set_mesh`` 方法来配置 mesh 信息，并将本 API 的 mesh 参数设置为 None。注意，如果您通过本 API 传递
      了 mesh 信息，传入的 mesh 会覆盖外部设置的 mesh。
    - **config** (dict，可选) - 用来指导并行化的配置。该配置是一个字典，键的值可以从``dp_config``，``mp_config`` 与
      ``pp_config`` 中进行选择，分别来指导数据并行、模型并行与流水线并行的并行化。一个合法的 config 可以是： ``{"dp_config":
      请参考本文档 dp_config 部分以获取更多内容, "mp_config": 请参考本文档 mp_config 部分以获取更多内容,
      "pp_config": 请参考本文档 pp_config 部分以获取更多内容}``。

      dp_config (dict)：指导数据并行的配置。该配置是一个字典，字典的键为 ``sharding_level`` 对应的值可以从 ``0/1/2/3`` 中选择。
      分别代表数据并行、sharding 并行 stage 1/2/3。一个合法的 dp_config 可以是：``{"sharding_level": 2}``.

      mp_config (dict)：指导模型并行的配置。该配置是一个字典，字典的键为 ``parallelize_plan`` 对应值仍然为一个字典，将标识的 Layer 的
      名字或一个参数的名字与对应的策略进行映射。注意：这里被标识的 Layer 的名字可以按照正则字符串的格式来书写。注意：如果将一个参数的名字与
      策略进行映射，该参数的名字必须以 weight 或者 bias 结尾。所有合法的策略包含：``ColWiseParallel``，``RowWiseParallel``，
      ``SequenceParallelBegin``，``SequenceParallelDisable``，``SequenceParallelEnable``，``SequenceParallelEnd``，
      ``PrepareLayerInput`` 和 ``PrepareLayerOutput``。一个合法的 mp_config 可以是： ``{"parallelize_plan":
      {"llama.embed_tokens": ColWiseParallel(), "llama.norm": SequenceParallelEnable(),
      "lm_head.weight": ColWiseParallel()}}``。

      pp_config (dict)：指导流水线并行的配置。该配置是一个字典，字典的键为 ``split_spec`` 与 ``global_spec`` （可选）。``split_spec``
      可以是一个字典或者是一个字符串。如果 ``split_spec`` 是一个字典，它将标识的 Layer 的名字与一个 ``SplitPoint`` 的值进行映射。
      注意：这里被标识的 Layer 的名字可以按照正则字符串的格式来书写。流水线并行将严格按照 ``split_spec`` 中指定的 Layer 进行切分。如果
      ``split_spec`` 是一个字符串，它会包含一系列 Layer 名字的前缀。流水线并行会根据流水线并行的规模自动在目标 Layer 集中切分模型。
      ``global_spec`` 是一个可选择的键，它的值是一个字符串，标明被标识的 Layer 包含需要在各个流水线层间被全局复制的全局张量。一些合法的
      pp_config 可以是：``{"split_spec": "llama.layers", "global_spec": "llama.global_layer"}`` 或者 ``{"split_spec":
      {"llama.layers.1": SplitPoint.END}}``。

返回
:::::::::
Model：一个具有分布式视角的 `Model` 对象。

Optimizer：一个具有分布式视角的 `Optimizer` 对象。如果用户为提供优化器对象，则返回 None。


代码示例
:::::::::

COPY-FROM: paddle.distributed.parallelize
