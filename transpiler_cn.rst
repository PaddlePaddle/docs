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
        
.. py:method:: transpile(trainer_id, program=None, pservers='127.0.0.1:6174', trainers=1, sync_mode=True, startup_program=None, current_endpoint='127.0.0.1:6174')
 
运行transpiler。

参数：
        - **trainer_id** （int）：当前训练操作的id，如果你有n个操作，id范围是0到n-1。
        - **program** （Program | None）：要transpile的程序，默认为 ``fluid.default_main_program（）`` 。
        - **pservers** （str）：逗号分隔的ip:port的字符串传给pserver列表。
        - **trainers** （int | str）：在pserver模式下这是 ``trainers`` 的数量，在 ``nccl2`` 模式下，这是一串 ``trainers endpoints`` 。
        - **sync_mode** （bool）：是否同步训练，默认为True。
        - **startup_program** （Program | None）：要运行的 ``startup_program`` ，默认为 ``fluid.default_main_program（）`` 。
        - **current_endpoint** （str）：需要在转换为 ``nccl2`` 分布式模式时传递当前端点。 在 ``pserver`` 模式下，不使用此参数。



.. py:method:: get_trainer_program(wait_port=True)

获取已编译的trainer程序。

返回：     trainer程序。

返回类型：   程序

.. py:method:: get_pserver_program(endpoint)

获取参数服务器端程序。

参数：
        - **endpoint** （str）：当前参数服务器端点。

返回：     当前参数服务器运行的程序。

返回类型：   程序

.. py:method:: get_pserver_programs(endpoint)

获取pserver端的主程序和启动程序以进行分布式训练。

参数：
        - **endpoint** （str）：当前pserver端点。
        
返回：     （main_program，startup_program），类型为 ``Program`` 。

返回类型：   tuple。

.. py:method:: get_startup_program(endpoint, pserver_program=None, startup_program=None)

不推荐使用。

获取当前参数服务器的启动程序。如果存在拆分为多个块的变量，则将对输入变量进行修改操作。

参数：
        - **endpoint** （str）：当前 ``pserver`` 端点。
        - **pserver_program** （Program）：不推荐使用，首先调用 ``get_pserver_program`` 。
        - **startup_program** （Program）：不推荐使用，应该在初始化时传递 ``startup_program`` 。

返回：     参数服务器端启动程序。

返回类型：   Program。





























