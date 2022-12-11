.. _api_guide_async_training:

############
分布式异步训练
############

Fluid 支持数据并行的分布式异步训练，API 使用 :code:`DistributeTranspiler` 将单机网络配置转换成可以多机执行的
:code:`pserver` 端程序和 :code:`trainer` 端程序。用户在不同的节点执行相同的一段代码，根据环境变量或启动参数，
可以执行对应的 :code:`pserver` 或 :code:`trainer` 角色。Fluid 异步训练只支持 pserver 模式，异步训练和 `同步训练 <../distributed/sync_training.html>`_ 的主要差异在于：异步训练每个 trainer 的梯度是单独更新到参数上的，
而同步训练是所有 trainer 的梯度合并之后统一更新到参数上，因此，同步训练和异步训练的超参数需要分别调节。

pserver 模式分布式异步训练
======================

API 详细使用方法参考 :ref:`cn_api_fluid_DistributeTranspiler` ，简单示例用法：

.. code-block:: python

    config = fluid.DistributeTranspilerConfig()
    # 配置策略 config
    config.slice_var_up = False
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(trainer_id,
                program=main_program,
                pservers="192.168.0.1:6174,192.168.0.2:6174",
                trainers=1,
                sync_mode=False)

以上参数说明请参考 `同步训练 <../distributed/sync_training.html>`_

需要注意的是：进行异步训练时，请修改 :code:`sync_mode` 的值

- :code:`sync_mode` ： 是否是同步训练模式，默认为 True，不传此参数也默认是同步训练模式，设置为 False 则为异步训练
