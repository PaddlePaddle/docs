.. _cn_api_fluid_DistributeTranspiler:

DistributeTranspiler
-------------------------------

.. py:class:: paddle.fluid.DistributeTranspiler (config=None)


该类可以把fluid program转变为分布式数据并行计算程序（distributed data-parallelism programs）,可以有Pserver和NCCL2两种模式。
当program在Pserver（全称：parameter server）模式下， ``main_program`` (主程序)转为使用一架远程parameter server(即pserver,参数服务器)来进行参数优化，并且优化图会被输入到一个pserver program中。
在NCCL2模式下，transpiler会在 ``startup_program`` 中附加一个 ``NCCL_ID`` 广播算子（broadcasting operators）来实现在该集群中所有工作结点共享 ``NCCL_ID`` 。
调用 ``transpile_nccl2`` 后， 你 **必须** 将 ``trainer_id`` , ``num_trainers`` 参数提供给 ``ParallelExecutor`` 来启动NCCL2分布式模式。




**代码示例**

.. code-block:: python

  import paddle.fluid as fluid
  x = fluid.layers.data(name='x', shape=[13], dtype='float32')
  y = fluid.layers.data(name='y', shape=[1], dtype='float32')
  y_predict = fluid.layers.fc(input=x, size=1, act=None)
  
  cost = fluid.layers.square_error_cost(input=y_predict, label=y)
  avg_loss = fluid.layers.mean(cost)
  
  sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
  sgd_optimizer.minimize(avg_loss)

  #pserver模式下
  pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
  trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
  current_endpoint = "192.168.0.1:6174"
  trainer_id = 0
  trainers = 4
  role = "PSERVER"

  t = fluid.DistributeTranspiler()
  t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
  if role == "PSERVER":
     pserver_program = t.get_pserver_program(current_endpoint)
     pserver_startup_program = t.get_startup_program(current_endpoint, pserver_program)
  elif role == "TRAINER":
     trainer_program = t.get_trainer_program()

  # nccl2模式下
  trainer_num = 2
  trainer_id = 0
  config = fluid.DistributeTranspilerConfig()
  config.mode = "nccl2"
  trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
  t = fluid.DistributeTranspiler(config=config)
  t.transpile(trainer_id=trainer_id, trainers=trainer_endpoints, current_endpoint="192.168.0.1:6174")
  exe = fluid.ParallelExecutor(
     loss_name=avg_loss.name,
     num_trainers=len(trainer_num,
     trainer_id=trainer_id
  )



.. py:method:: transpile(trainer_id, program=None, pservers='127.0.0.1:6174', trainers=1, sync_mode=True, startup_program=None, current_endpoint='127.0.0.1:6174')

该方法可以运行该transpiler（转译器）。转译输入程序。

参数:
  - **trainer_id** (int) – 当前Trainer worker的id, 如果有n个Trainer worker, id 取值范围为0 ~ n-1
  - **program** (Program|None) – 待transpile（转译）的program, 缺省为 ``fluid.default_main_program()``
  - **startup_program** (Program|None) - 要转译的 ``startup_program`` ,默认为 ``fluid.default_startup_program()``
  - **pservers** (str) – 内容为Pserver列表的字符串，格式为：按逗号区分不同的Pserver，每个Pserver的格式为 *ip地址:端口号*
  - **trainers** (int|str) – 在Pserver模式下，该参数指Trainer机的个数；在nccl2模式下，它是一个内容为Trainer终端列表的字符串
  - **sync_mode** (bool) – 是否做同步训练(synchronous training), 默认为True
  - **startup_program** (Program|None) – 待transpile（转译）的startup_program，默认为 ``fluid.default_main_program()``
  - **current_endpoint** (str) – 当需要把program转译（transpile）至NCCL2模式下时，需要将当前endpoint（终端）传入该参数。Pserver模式不使用该参数

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    transpiler = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id=0,
        pservers="127.0.0.1:7000,127.0.0.1:7001",
        trainers=2,
        sync_mode=False,
        current_endpoint="127.0.0.1:7000")



.. py:method:: get_trainer_program(wait_port=True)


该方法可以得到Trainer侧的program。

返回: Trainer侧的program

返回类型: Program

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    #this is an example, find available endpoints in your case
    pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
    trainer_id = 0
    trainers = 4
    t = fluid.DistributeTranspiler()
    t.transpile(trainer_id, trainers=trainers, pservers=pserver_endpoints)
    trainer_program = t.get_trainer_program()


.. py:method:: get_pserver_program(endpoint)


该方法可以得到Pserver（参数服务器）侧的程序

参数:
  - **endpoint** (str) – 当前Pserver终端

返回: 当前Pserver需要执行的program

返回类型: Program

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    #this is an example, find available endpoints in your case
    pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
    current_endpoint = "192.168.0.1:6174"
    trainer_id = 0
    trainers = 4
    t = fluid.DistributeTranspiler()
    t.transpile(
         trainer_id, pservers=pserver_endpoints, trainers=trainers)
    pserver_program = t.get_pserver_program(current_endpoint)


.. py:method:: get_pserver_programs(endpoint)


该方法可以得到Pserver侧用于分布式训练的 ``main_program`` 和 ``startup_program`` 。

参数:
  - **endpoint** (str) – 当前Pserver终端

返回: (main_program, startup_program), “Program”类型的元组

返回类型: tuple

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    #this is an example, find available endpoints in your case
    pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
    current_endpoint = "192.168.0.1:6174"
    trainer_id = 0
    trainers = 4
    t = fluid.DistributeTranspiler()
    t.transpile(
         trainer_id, pservers=pserver_endpoints, trainers=trainers)
    pserver_program, pserver_startup_program = t.get_pserver_programs(current_endpoint)



.. py:method:: get_startup_program(endpoint, pserver_program=None, startup_program=None)


**该函数已停止使用**
获取当前Pserver的startup_program，如果有多个被分散到不同blocks的变量，则修改operator的输入变量。

参数:
  - **endpoint** (str) – 当前Pserver终端
  - **pserver_program** (Program) – 已停止使用。 先调用get_pserver_program
  - **startup_program** (Program) – 已停止使用。应在初始化时传入startup_program

返回: Pserver侧的startup_program

返回类型: Program

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
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
     





