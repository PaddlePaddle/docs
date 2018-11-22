.. _cn_api_fluid_transpiler_DistributeTranspiler:

DistributeTranspiler
>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.transpiler.DistributeTranspiler(config=None)

分布式Transpiler

将fluid程序转换为分布式数据并行程序。支持两种模式： ``pserver`` 模式和 ``nccl2`` 模式。

在 ``pserver`` 模式下，主程序将被转换使用远程参数服务器进行参数优化。优化图将被放入参数服务器程序中。

在 ``nccl2`` 模式下，转换器将在 ``startup_program`` 中附加 ``NCCL_ID`` 传播操作，以在作业节点之间共享 ``NCCL_ID`` 。 在调用 ``transpile_nccl2`` 之后，您必须将 ``trainer_id`` 和 ``num_trainers`` 参数传递给 ``ParallelExecutor`` 以启用 ``NCCL2`` 分布式模式。

**示例代码**

..  code-block:: python

        # pserver模式
        pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
        trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
        current_endpoint = "192.168.0.1:6174"
        trainer_id = 0
        trainers = 4
        role = os.getenv("PADDLE_TRAINING_ROLE")

        t = fluid.DistributeTranspiler()
        t.transpile(
             trainer_id, pservers=pserver_endpoints, trainers=trainers)
        if role == "PSERVER":
             pserver_program = t.get_pserver_program(current_endpoint)
             pserver_startup_program = t.get_startup_program(current_endpoint,
                                                             pserver_program)
        elif role == "TRAINER":
             trainer_program = t.get_trainer_program()

        # nccl2模式
        config = fluid.DistributeTranspilerConfig()
        config.mode = "nccl2"
        t = fluid.DistributeTranspiler(config=config)
        t.transpile(trainer_id, workers=workers, current_endpoint=curr_ep)
        exe = fluid.ParallelExecutor(
            use_cuda,
            loss_name=loss_var.name,
            num_trainers=len(trainers.split(",)),
            trainer_id=trainer_id
        )
        
        
