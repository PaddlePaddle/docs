#################
 fluid.transpiler
#################



.. _cn_api_fluid_transpiler_DistributeTranspiler:

DistributeTranspiler
-------------------------------

.. py:class:: paddle.fluid.transpiler.DistributeTranspiler (config=None)


该类可以把fluid program转变为分布式数据并行计算程序（distributed data-parallelism programs）,可以有Pserver和NCCL2两种模式。
当program在Pserver（全称：parameter server）模式下， ``main_program`` (主程序)转为使用一架远程parameter server(即pserver,参数服务器)来进行参数优化，并且优化图会被输入到一个pserver program中。
在NCCL2模式下，transpiler会在 ``startup_program`` 中附加一个 ``NCCL_ID`` 广播算子（broadcasting operators）来实现在该集群中所有工作结点共享``NCCL_ID`` 。
调用 ``transpile_nccl2`` 后， 你 **必须** 将 ``trainer_id`` , ``num_trainers`` 参数提供给 ``ParallelExecutor`` 来启动NCCL2分布式模式。 




**代码示例**

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

    transpiler = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id=0,
        pservers="127.0.0.1:7000,127.0.0.1:7001",
        trainers=2,
        sync_mode=False,
        current_endpoint="127.0.0.1:7000")


.. py:method:: get_trainer_program(wait_port=True)


该方法可以得到Trainer侧的program。

返回:    Trainer侧的program

返回类型:    Program

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


.. py:method:: get_pserver_program(endpoint)


该方法可以得到Pserver（参数服务器）侧的程序
 
参数:    
    - **endpoint** (str) – 当前Pserver终端
 
返回:    当前Pserver需要执行的program

返回类型:    Program

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


.. py:method:: get_pserver_programs(endpoint)


该方法可以得到Pserver侧用于分布式训练的 ``main_program`` 和 ``startup_program`` 。

参数:    
    - **endpoint** (str) – 当前Pserver终端

返回:    (main_program, startup_program), “Program”类型的元组

返回类型:    tuple 
 
 
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


.. py:method:: get_startup_program(endpoint, pserver_program=None, startup_program=None)


**该函数已停止使用**
获取当前Pserver的startup_program，如果有多个被分散到不同blocks的变量，则修改operator的输入变量。

参数:    
    - **endpoint** (str) – 当前Pserver终端
    - **pserver_program** (Program) – 已停止使用。 先调用get_pserver_program
    - **startup_program** (Program) – 已停止使用。应在初始化时传入startup_program

返回:    Pserver侧的startup_program

返回类型:    Program

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



.. _cn_api_fluid_transpiler_DistributeTranspilerConfig:

DistributeTranspilerConfig
-------------------------------

.. py:class:: paddle.fluid.transpiler.DistributeTranspilerConfig

.. py:method:: slice_var_up (bool)

为Pserver将张量切片, 默认为True

.. py:method:: split_method (PSDispatcher)

可使用 RoundRobin 或者 HashName

注意: 尝试选择最佳方法来达到负载均衡。


.. py:attribute:: min_block_size (int)

最小数据块的大小

注意: 根据：https://github.com/PaddlePaddle/Paddle/issues/8638#issuecomment-369912156 , 当数据块大小超过2MB时，我们可以有效地使用带宽。如果你想更改它，请详细查看slice_variable函数。

**代码示例**

.. code-block:: python

        config = fluid.DistributeTranspilerConfig()
        config.slice_var_up = True



.. _cn_api_fluid_transpiler_HashName:

HashName
-------------------------------

.. py:class:: paddle.fluid.transpiler.HashName(pserver_endpoints)

使用 python ``Hash()`` 函数将变量名散列到多个pserver终端。

参数:
  - **pserver_endpoints** (list) - endpoint （ip:port）的 list 

**代码示例**

.. code-block:: python

          pserver_endpoints = [“127.0.0.1:6007”, “127.0.0.1:6008”]
          vars = [“var1”,”var2”,”var3”,”var4”,”var5”]
          rr = RoundRobin(pserver_endpoints)
          rr.dispatch(vars)




.. _cn_api_fluid_transpiler_memory_optimize:

memory_optimize
-------------------------------

.. py:function:: paddle.fluid.transpiler.memory_optimize(input_program, skip_opt_set=None, print_log=False, level=0, skip_grads=False)

历史遗留内存优化策略，通过在不同operators之间重用可变内存来减少总内存消耗。
用一个简单的例子来解释该算法：

c = a + b  # 假设此处是最后一次使用a
d = b * c

因为在“c = a + b”之后将不再使用a，并且a和d的大小相同，所有我们可以使用变量a来替换变量d，即实际上我们可以将上面的代码优化为如下所示：

c = a + b
a = b * c

请注意，在这个历史遗留设计中，我们使用变量a直接替换d，这意味着在调用此API之后，某些变量可能会消失，而某些变量可能会保留非预期值，如在上面的例子中，实际上执行代码后a保持d的值。

因此，为了防止重要变量在优化中被重用/删除，我们提供skip_opt_set用于指定变量白名单。
skip_opt_set中的变量不受memory_optimize API的影响。

注意：
此API已弃用，请避免在新代码中使用它。
不支持会创建子块的运算符，如While，IfElse等。

参数:
  - **input_program** (str) – 输入Program。
  - **skip_opt_set** (set) – set中的vars将不被内存优化。
  - **print_log** (bool) – 是否打印debug日志。
  - **level** (int) - 0或1，0代表我们仅在a.size == b.size时用b替换a，1代表我们可以在a.size <= b.size时用b替换a

返回: None

**代码示例**

.. code-block:: python

          import paddle.fluid as fluid
          main_prog = fluid.Program()
          startup_prog = fluid.Program()
           
          place = fluid.CPUPlace()
          exe = fluid.Executor(place)
           
          exe.run(startup_prog)
          fluid.memory_optimize(main_prog)




.. _cn_api_fluid_transpiler_release_memory:

release_memory
-------------------------------

.. py:function:: paddle.fluid.transpiler.release_memory(input_program, skip_opt_set=None) 


该函数可以调整输入program，插入 ``delete_op`` 删除算子，提前删除不需要的变量。
改动是在变量本身上进行的。

**提醒** : 该API还在试验阶段，会在后期版本中删除。不建议用户使用。

参数:
    - **input_program** (Program) – 在此program中插入 ``delete_op`` 
    - **skip_opt_set** (set) – 在内存优化时跳过的变量的集合

返回: None

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        # 构建网络
        # ...
        
        # 已弃用的API
        fluid.release_memory(fluid.default_main_program())











.. _cn_api_fluid_transpiler_RoundRobin:

RoundRobin
-------------------------------

.. py:class:: paddle.fluid.transpiler.RoundRobin(pserver_endpoints)

使用 ``RondRobin`` 方法将变量分配给服务器端点。

`RondRobin <https://en.wikipedia.org/wiki/Round-robin_scheduling>`_

参数:
  - **pserver_endpoints** (list) - endpoint （ip:port）的 list 
 
**代码示例**

.. code-block:: python

          pserver_endpoints = [“127.0.0.1:6007”, “127.0.0.1:6008”]
          vars = [“var1”,”var2”,”var3”,”var4”,”var5”]
          rr = RoundRobin(pserver_endpoints)
          rr.dispatch(vars)




