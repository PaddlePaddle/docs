.. _cn_api_fluid_transpiler_DistributeTranspiler:

DistributeTranspiler
-------------------------------


.. py:class:: paddle.fluid.transpiler.DistributeTranspiler (config=None)





该类可以把fluid program转变为分布式数据并行计算的program，有PServer和NCCL2两种模式。
在Pserver（全称：parameter server）模式下，通过 ``transpile`` 将用于单机训练的 ``program``  转译为可用于parameter server的分布式架构(即PServer，参数服务器)来进行训练的program。
在NCCL2模式下，通过 ``transpile`` 将用于单机训练的 ``program``  转译为可用于NCCL2的分布式架构来进行训练的program。在NCCL2模式下，transpiler会在 ``startup_program`` 中附加一个 ``NCCL_ID`` 广播算子（broadcasting operators）来实现在该集群中所有工作结点共享``NCCL_ID``。调用 ``transpile_nccl2`` 后，你 **必须** 将 ``trainer_id`` , ``num_trainers`` 参数提供给 ``Executor`` 来启动NCCL2分布式模式。


参数
::::::::::::

        - **config** （DistributeTranspilerConfig） DistributeTranspiler属性配置实例，定义了program转变所需要的属性，请参考：`DistributeTranspilerConfig` 相关文档。

返回
::::::::::::
初始化后的DistributeTranspiler实例

返回类型
::::::::::::
实例（DistributeTranspiler）


代码示例
::::::::::::

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    # pserver 模式下
    pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
    trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
    current_endpoint = "192.168.0.1:6174"
    trainer_id = 0
    trainers = 4
    role = "PSERVER"
    t = fluid.DistributeTranspiler()
    t.transpile(
         trainer_id, pservers=pserver_endpoints, trainers=trainers)
    if role == "PSERVER":
         pserver_program = t.get_pserver_program(current_endpoint)
         pserver_startup_program = t.get_startup_program(current_endpoint,
                                                        pserver_program)
    elif role == "TRAINER":
         trainer_program = t.get_trainer_program()

    # nccl2 模式下
    trainer_num = 2
    trainer_id = 0
    config = fluid.DistributeTranspilerConfig()
    config.mode = "nccl2"
    trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(trainer_id=trainer_id, trainers=trainer_endpoints, current_endpoint="192.168.0.1:6174")
    exe = fluid.ParallelExecutor(
        use_cuda=True,
        loss_name=avg_loss.name,
        num_trainers=trainer_num,
        trainer_id=trainer_id
    )



方法
::::::::::::
transpile(trainer_id, program=None, pservers='127.0.0.1:6174', trainers=1, sync_mode=True, startup_program=None, current_endpoint='127.0.0.1:6174')
'''''''''

通过此方法，可根据用户配置将单机的program转换为当前节点可用的数据并行的分布式program。

**参数**
    
    - **trainer_id** (int) – 当前Trainer worker的id，如果有n个Trainer worker, id 取值范围为0 ~ n-1
    - **program** (Program|None) – 待transpile（转译）的main program，默认为 ``fluid.default_main_program()`` 
    - **pservers** (str) – 内容为Pserver列表的字符串，格式为：按逗号区分不同的Pserver，每个Pserver的格式为 *ip地址：端口号* 
    - **trainers** (int|str) – 在Pserver模式下，该参数指Trainer机的个数；在nccl2模式下，它是一个内容为Trainer终端列表的字符串
    - **sync_mode** (bool) – 是否做同步训练(synchronous training)，默认为True
    - **startup_program** (Program|None) – 待transpile（转译）的startup program，默认为 ``fluid.default_startup_program()``
    - **current_endpoint** (str) – 当需要把program转译（transpile）至NCCL2模式时，需要将当前endpoint（终端）传入该参数。PServer模型下，当用户需要使用增量训练时，必须要指定该参数。

**返回**
None


**代码示例**

.. code-block:: python

    transpiler = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id=0,
        pservers="127.0.0.1:7000,127.0.0.1:7001",
        trainers=2,
        sync_mode=False,
        current_endpoint="127.0.0.1:7000")


get_trainer_program(wait_port=True)
'''''''''


该方法可以得到Trainer侧的program。Trainer侧的program相较于原始的单机执行的program，主要有以下不同：

     - 删除了参数更新optimizer相关op，参数的更新由Pserver（参数服务器）执行
     - 在每个参数的反向梯度计算op后，添加了 ``Send_op`` 与 ``Recv_op``，用于发送参数的梯度与接受更新后的参数

**参数**

     - **wait_port** (bool，默认值True) - 是否等待参数服务器准备就绪后再返回program

**返回**
    Trainer侧的program

**返回类型**
    Program

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        # 这是一个示例，请根据你的情况更改endpoint
        pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
        trainer_id = 0
        trainers = 4
        t = fluid.DistributeTranspiler()
        t.transpile(trainer_id, trainers=trainers, pservers=pserver_endpoints)
        trainer_program = t.get_trainer_program()


get_pserver_program(endpoint)
'''''''''


该方法可以得到Pserver（参数服务器）侧的program。Pserver侧的program相较于原始的单机执行的program，主要有以下不同：
     
     - 仅包含参数更新optimizer相关op，与分布式通信相关op
     - 0号block仅包含变量的定义及 ``listen_and_serv_op`` 
     - Pserver为每个需要进行更新的参数新建了一个独立的block
 
**参数**
    
    - **endpoint** (str) – 当前Pserver终端
 
**返回**
    当前Pserver需要执行的program

**返回类型**
    Program

**代码示例**

.. code-block:: python

          import paddle.fluid as fluid
          # 这是一个示例，请根据你的情况更改endpoint
          pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
          current_endpoint = "192.168.0.1:6174"
          trainer_id = 0
          trainers = 4
          t = fluid.DistributeTranspiler()
          t.transpile(
               trainer_id, pservers=pserver_endpoints, trainers=trainers)
          pserver_program = t.get_pserver_program(current_endpoint)


get_pserver_programs(endpoint)
'''''''''


该方法可以得到Pserver侧用于分布式训练的 ``main_program`` 和 ``startup_program``。该函数返回的 ``main_program`` 与函数 ``get_pserver_program`` 的返回值一致。

**参数**
    
    - **endpoint** (str) – 当前Pserver终端

**返回**
    (main_program, startup_program), “Program”类型的元组

**返回类型**
    tuple 
 
 
**代码示例**

.. code-block:: python

          import paddle.fluid as fluid
          # 这是一个示例，请根据你的情况更改endpoint
          pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
          current_endpoint = "192.168.0.1:6174"
          trainer_id = 0
          trainers = 4
          t = fluid.DistributeTranspiler()
          t.transpile(
               trainer_id, pservers=pserver_endpoints, trainers=trainers)
          pserver_program, pserver_startup_program = t.get_pserver_programs(current_endpoint)


get_startup_program(endpoint, pserver_program=None, startup_program=None)
'''''''''


**该函数已停止使用**
获取当前Pserver的startup_program，如果有多个被分散到不同blocks的变量，则修改operator的输入变量。

**参数**
    
    - **endpoint** (str) – 当前Pserver终端
    - **pserver_program** (Program) – 已停止使用。先调用get_pserver_program
    - **startup_program** (Program) – 已停止使用。应在初始化时传入startup_program

**返回**
    Pserver侧的startup_program

**返回类型**
    Program

**代码示例**

.. code-block:: python

          pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
          trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
          current_endpoint = "192.168.0.1:6174"
          trainer_id = 0
          trainers = 4
           
          t = fluid.DistributeTranspiler()
          t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
          pserver_program = t.get_pserver_program(current_endpoint)
          pserver_startup_program = t.get_startup_program(current_endpoint,
                                                          pserver_program)



