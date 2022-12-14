.. _cn_api_distributed_fleet_Fleet:

Fleet
-------------------------------


.. py:class:: paddle.distributed.fleet.Fleet

Fleet 是飞桨分布式训练统一 API，只需要 import fleet 并简单初始化后即可快速开始使用飞桨大规模分布式训练


方法
::::::::::::
init(role_maker=None, is_collective=False, strategy=None)
'''''''''

使用 RoleMaker 或其他配置初始化 fleet。


**参数**

    - **role_maker** (RoleMakerBase) 已初始化好的 PaddleCloudRoleMaker 或 UserDefineRoleMaker
    - **is_collective** (bool) 在未指定 role_maker 的情况下，可由 init 方法自行初始化 RoleMaker, is_collective 为 True 则按照 collective 模式进行创建，is_collective=False 则按照 ParameterServer 模式进行创建
    - **strategy** (DistributedStrategy)：分布式训练的额外属性。详情请参阅 paddle.distributed.fleet.DistributedStrategy。默认值：None。

**返回**
None


**代码示例 1**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()

**代码示例 2**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init(is_collective=True)

**代码示例 3**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    role = fleet.PaddleCloudRoleMaker()
    fleet.init(role)

**代码示例 4**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    strategy = fleet.DistributedStrategy()
    fleet.init(strategy=strategy)


is_first_worker()
'''''''''

返回当前节点是否为第一个`worker`节点，判断当前 worker_index 是否为 0，如果为 0 则返回 True，否则返回 False。

**返回**
True/False


**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.is_first_worker()



worker_index()
'''''''''

返回当前节点的编号，每个`worker`节点被分配[0, worker_num-1]内的唯一的编码 ID

**返回**
int


**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.worker_index()


worker_num()
'''''''''

返回当前全部训练节点中`workjer`节点的个数

**返回**
int

**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.worker_num()


is_worker()
'''''''''

返回当前节点是否为`worker`节点

**返回**
True/False

**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.is_worker()


worker_endpoints(to_string=False)
'''''''''

返回全部 worker 节点的 ip 及端口信息

**返回**
list/string

**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.worker_endpoints()


server_num()
'''''''''

**注意：**

  **该参数只在 ParameterServer 模式下生效**


返回当前全部 Server 节点的个数

**返回**
int

**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.server_num()


server_index()
'''''''''


**注意：**

  **该参数只在 ParameterServer 模式下生效**


返回当前节点的编号，每个`server`节点被分配[0, server_num-1]内的唯一的编码 ID

**返回**
int


**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.server_index()


server_endpoints(to_string=False)
'''''''''


**注意：**

  **该参数只在 ParameterServer 模式下生效**


返回全部 server 节点的 ip 及端口信息

**返回**
list/string

**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.server_endpoints()


is_server()
'''''''''


**注意：**

  **该参数只在 ParameterServer 模式下生效**


返回当前节点是否为`server`节点

**返回**
True/False

**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.is_server()


barrier_worker()
'''''''''

调用集合通信功能，强制要求所有的 worker 在此处相互等待一次

**返回**
无

**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.barrier_worker()


init_worker()
'''''''''

worker 节点在训练前的初始化，包括通信模块，参数同步等

**返回**
无

**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.init_worker()


init_server(*args, **kwargs)
'''''''''

server 节点的初始化，包括 server 端参数初始化，模型加载等

**返回**
无

**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.init_server()


run_server()
'''''''''

server 节点的运行，此命令会将 ParameterServer 的进程启动并常驻直至训练结束

**返回**
无

**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.init_server()
    fleet.run_server()


stop_worker()
'''''''''

停止当前正在运行的 worker 节点

**返回**
无

**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    fleet.init()
    fleet.init_worker()
    "..."
    fleet.stop_worker()


save_inference_model(executor, dirname, feeded_var_names, target_vars, main_program=None, export_for_deployment=True)
'''''''''

修剪指定的 ``main_program`` 以构建一个专门用于预测的 ``Inference Program`` （ ``Program`` 含义详见 :ref:`api_guide_Program` ）。所得到的 ``Inference Program`` 及其对应的所>有相关参数均被保存到 ``dirname`` 指定的目录中。


**参数**

  - **executor** (Executor) –  用于保存预测模型的 ``executor``，详见 :ref:`api_guide_executor` 。
  - **dirname** (str) – 指定保存预测模型结构和参数的文件目录。
  - **feeded_var_names** (list[str]) – 字符串列表，包含着 Inference Program 预测时所需提供数据的所有变量名称（即所有输入变量的名称）。
  - **target_vars** (list[Tensor]) – ``Tensor`` （详见 :ref:`api_guide_Program` ）类型列表，包含着模型的所有输出变量。通过这些输出变量即可得到模型的预测结果。
  - **main_program** (Program，可选) – 通过该参数指定的 ``main_program`` 可构建一个专门用于预测的 ``Inference Program``。若为 None，则使用全局默认的  ``_main_program_`` 。>默认值为 None。
  - **export_for_deployment** (bool，可选) – 若为 True，则 ``main_program`` 指定的 Program 将被修改为只支持直接预测部署的 Program。否则，将存储更多的信息，方便优化和再训练。目前
只支持设置为 True，且默认值为 True。


**返回**
无

**代码示例**

.. code-block:: text

    import paddle
    paddle.enable_static()
    import paddle.distributed.fleet as fleet

    fleet.init()

    # build net
    # loss = Net()
    # fleet.distributed_optimizer(...)

    exe = paddle.static.Executor(paddle.CPUPlace())
    fleet.save_inference_model(exe, "dirname", ["feed_varname"], [loss], paddle.static.default_main_program())


save_persistables(executor, dirname, main_program=None)
'''''''''


保存全量模型参数

**参数**

 - **executor**  (Executor) – 用于保存持久性变量的 ``executor``，详见 :ref:`api_guide_executor` 。
 - **dirname**  (str) – 用于储存持久性变量的文件目录。
 - **main_program**  (Program，可选) – 需要保存持久性变量的 Program（ ``Program`` 含义详见 :ref:`api_guide_Program` ）。如果为 None，则使用 default_main_Program。默认值为 None>。

**返回**
无

**代码示例**

.. code-block:: text

    import paddle
    paddle.enable_static()
    import paddle.distributed.fleet as fleet

    fleet.init()

    # build net
    # fleet.distributed_optimizer(...)

    exe = paddle.static.Executor(paddle.CPUPlace())
    fleet.save_persistables(exe, "dirname", paddle.static.default_main_program())


distributed_optimizer(optimizer, strategy=None)
'''''''''

基于分布式布式并行策略进行模型的拆分及优化。

**参数**

 - **optimizer**  (optimizer) – paddle 定义的优化器。
 - **strategy**  (DistributedStrategy) – 分布式优化器的额外属性。建议在 fleet.init()创建。这里的仅仅是为了兼容性。如果这里的参数 strategy 不是 None，则它将覆盖在 fleet.init()创建的 DistributedStrategy，并在后续的分布式训练中生效。

**代码示例**

.. code-block:: python

    import paddle
    paddle.enable_static()
    import paddle.distributed.fleet as fleet
    fleet.init(is_collective=True)
    strategy = fleet.DistributedStrategy()
    optimizer = paddle.optimizer.SGD(learning_rate=0.001)
    optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)


distributed_model(model)
'''''''''

**注意：**

  **1. 该 API 只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

返回分布式数据并行模型。

**参数**

    model (Layer) - 用户定义的模型，此处模型是指继承动态图 Layer 的网络。

**返回**
分布式数据并行模型，该模型同样继承动态图 Layer。


**代码示例**

.. code-block:: python


    # 这个示例需要由 fleetrun 启动，用法为：
    # fleetrun --gpus=0,1 example.py
    # 脚本 example.py 中的代码是下面这个示例。

    import paddle
    import paddle.nn as nn
    from paddle.distributed import fleet

    class LinearNet(nn.Layer):
        def __init__(self):
            super().__init__()
            self._linear1 = nn.Linear(10, 10)
            self._linear2 = nn.Linear(10, 1)

        def forward(self, x):
            return self._linear2(self._linear1(x))

    # 1. initialize fleet environment
    fleet.init(is_collective=True)

    # 2. create layer & optimizer
    layer = LinearNet()
    loss_fn = nn.MSELoss()
    adam = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=layer.parameters())

    # 3. get data_parallel model using fleet
    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)

    # 4. run layer
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    print("loss:", loss.numpy())

    loss.backward()

    adam.step()
    adam.clear_grad()

state_dict()
'''''''''

**注意：**

  **1. 该 API 只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

以 ``dict`` 返回当前 ``optimizer`` 使用的所有 Tensor。比如对于 Adam 优化器，将返回 beta1, beta2, momentum 等 Tensor。

**返回**
dict，当前 ``optimizer`` 使用的所有 Tensor。


**代码示例**

.. code-block:: python

    # 这个示例需要由 fleetrun 启动，用法为：
    # fleetrun --gpus=0,1 example.py
    # 脚本 example.py 中的代码是下面这个示例。

    import numpy as np
    import paddle
    from paddle.distributed import fleet

    fleet.init(is_collective=True)

    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.to_tensor(value)

    layer = paddle.nn.Linear(13, 5)
    adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)
    state_dict = adam.state_dict()


set_state_dict(state_dict)
'''''''''

**注意：**

  **1. 该 API 只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

加载 ``optimizer`` 的 Tensor 字典给当前 ``optimizer`` 。

**返回**
None


**代码示例**

.. code-block:: python

    # 这个示例需要由 fleetrun 启动，用法为：
    # fleetrun --gpus=0,1 example.py
    # 脚本 example.py 中的代码是下面这个示例。

    import numpy as np
    import paddle
    from paddle.distributed import fleet

    fleet.init(is_collective=True)

    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.to_tensor(value)

    layer = paddle.nn.Linear(13, 5)
    adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)
    state_dict = adam.state_dict()
    paddle.save(state_dict, "paddle_dy")
    para_state_dict = paddle.load( "paddle_dy")
    adam.set_state_dict(para_state_dict)


set_lr(value)
'''''''''

**注意：**

  **1. 该 API 只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

手动设置当前 ``optimizer`` 的学习率。

**参数**

    value (float) - 需要设置的学习率的值。

**返回**
None


**代码示例**

.. code-block:: python

    # 这个示例需要由 fleetrun 启动，用法为：
    # fleetrun --gpus=0,1 example.py
    # 脚本 example.py 中的代码是下面这个示例。

    import numpy as np
    import paddle
    from paddle.distributed import fleet

    fleet.init(is_collective=True)

    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.to_tensor(value)

    layer = paddle.nn.Linear(13, 5)
    adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)

    lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
    for i in range(5):
        adam.set_lr(lr_list[i])
        lr = adam.get_lr()
        print("current lr is {}".format(lr))
    # Print:
    #    current lr is 0.2
    #    current lr is 0.3
    #    current lr is 0.4
    #    current lr is 0.5
    #    current lr is 0.6


get_lr()
'''''''''

**注意：**

  **1. 该 API 只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

获取当前步骤的学习率。

**返回**
float，当前步骤的学习率。



**代码示例**

.. code-block:: python

    # 这个示例需要由 fleetrun 启动，用法为：
    # fleetrun --gpus=0,1 example.py
    # 脚本 example.py 中的代码是下面这个示例。

    import numpy as np
    import paddle
    from paddle.distributed import fleet

    fleet.init(is_collective=True)

    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.to_tensor(value)

    layer = paddle.nn.Linear(13, 5)
    adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)

    lr = adam.get_lr()
    print(lr) # 0.01


step()
'''''''''

**注意：**

  **1. 该 API 只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

执行一次优化器并进行参数更新。

**返回**
None。


**代码示例**

.. code-block:: python

    # 这个示例需要由 fleetrun 启动，用法为：
    # fleetrun --gpus=0,1 example.py
    # 脚本 example.py 中的代码是下面这个示例。

    import paddle
    import paddle.nn as nn
    from paddle.distributed import fleet

    class LinearNet(nn.Layer):
        def __init__(self):
            super().__init__()
            self._linear1 = nn.Linear(10, 10)
            self._linear2 = nn.Linear(10, 1)

        def forward(self, x):
            return self._linear2(self._linear1(x))

    # 1. initialize fleet environment
    fleet.init(is_collective=True)

    # 2. create layer & optimizer
    layer = LinearNet()
    loss_fn = nn.MSELoss()
    adam = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=layer.parameters())

    # 3. get data_parallel model using fleet
    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)

    # 4. run layer
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    print("loss:", loss.numpy())

    loss.backward()

    adam.step()
    adam.clear_grad()


clear_grad()
'''''''''

**注意：**

  **1. 该 API 只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**


清除需要优化的参数的梯度。

**返回**
None。


**代码示例**

.. code-block:: python

    # 这个示例需要由 fleetrun 启动，用法为：
    # fleetrun --gpus=0,1 example.py
    # 脚本 example.py 中的代码是下面这个示例。

    import paddle
    import paddle.nn as nn
    from paddle.distributed import fleet

    class LinearNet(nn.Layer):
        def __init__(self):
            super().__init__()
            self._linear1 = nn.Linear(10, 10)
            self._linear2 = nn.Linear(10, 1)

        def forward(self, x):
            return self._linear2(self._linear1(x))

    # 1. initialize fleet environment
    fleet.init(is_collective=True)

    # 2. create layer & optimizer
    layer = LinearNet()
    loss_fn = nn.MSELoss()
    adam = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=layer.parameters())

    # 3. get data_parallel model using fleet
    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)

    # 4. run layer
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    print("loss:", loss.numpy())

    loss.backward()

    adam.step()
    adam.clear_grad()


minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None)
'''''''''


属性
::::::::::::
util
'''''''''
