.. _cn_api_distributed_fleet_Fleet:

Fleet
-------------------------------


.. py:class:: paddle.distributed.fleet.Fleet




.. py:method:: init(role_maker=None, is_collective=False)


.. py:method:: is_first_worker()


.. py:method:: worker_index()


.. py:method:: worker_num()


.. py:method:: is_worker()


.. py:method:: worker_endpoints(to_string=False)


.. py:method:: server_num()


.. py:method:: server_index()


.. py:method:: server_endpoints(to_string=False)


.. py:method:: is_server()


.. py:method:: barrier_worker()


.. py:method:: init_worker()


.. py:method:: init_server(*args, **kwargs)


.. py:method:: run_server()


.. py:method:: stop_worker()


.. py:method:: save_inference_model(executor, dirname, feeded_var_names, target_vars, main_program=None, export_for_deployment=True)


.. py:method:: save_persistables(executor, dirname, main_program=None)


.. py:method:: distributed_optimizer(optimizer, strategy=None)


.. py:method:: distributed_model(model)

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

返回分布式数据并行模型。

参数：
    model (Layer) - 用户定义的模型，此处模型是指继承动态图Layer的网络。

返回：分布式数据并行模型，该模型同样继承动态图Layer。


**代码示例**

.. code-block:: python


    # 这个示例需要由fleetrun启动, 用法为:
    # fleetrun --gpus=0,1 example.py
    # 脚本example.py中的代码是下面这个示例.

    import paddle
    import paddle.nn as nn
    from paddle.distributed import fleet

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear1 = nn.Linear(10, 10)
            self._linear2 = nn.Linear(10, 1)

        def forward(self, x):
            return self._linear2(self._linear1(x))

    # 1. enable dynamic mode
    paddle.disable_static()

    # 2. initialize fleet environment
    fleet.init(is_collective=True)

    # 3. create layer & optimizer
    layer = LinearNet()
    loss_fn = nn.MSELoss()
    adam = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=layer.parameters())

    # 4. get data_parallel model using fleet
    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)

    # 5. run layer
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    print("loss:", loss.numpy())

    loss = dp_layer.scale_loss(loss)
    loss.backward()
    dp_layer.apply_collective_grads()

    adam.step()
    adam.clear_grad()

.. py:method:: state_dict()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

以 ``dict`` 返回当前 ``optimizer`` 使用的所有Tensor 。比如对于Adam优化器，将返回 beta1, beta2, momentum 等Tensor。

返回：dict, 当前 ``optimizer`` 使用的所有Tensor。


**代码示例**

.. code-block:: python

    # 这个示例需要由fleetrun启动, 用法为:
    # fleetrun --gpus=0,1 example.py
    # 脚本example.py中的代码是下面这个示例.

    import numpy as np
    import paddle
    from paddle.distributed import fleet

    paddle.disable_static()
    fleet.init(is_collective=True)

    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.fluid.dygraph.to_variable(value)

    layer = paddle.nn.Linear(13, 5)
    adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)
    state_dict = adam.state_dict()


.. py:method:: set_state_dict(state_dict)

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

加载 ``optimizer`` 的Tensor字典给当前 ``optimizer`` 。

返回：None


**代码示例**

.. code-block:: python

    # 这个示例需要由fleetrun启动, 用法为:
    # fleetrun --gpus=0,1 example.py
    # 脚本example.py中的代码是下面这个示例.

    import numpy as np
    import paddle
    from paddle.distributed import fleet

    paddle.disable_static()
    fleet.init(is_collective=True)

    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.fluid.dygraph.to_variable(value)

    layer = paddle.nn.Linear(13, 5)
    adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)
    state_dict = adam.state_dict()
    paddle.framework.save(state_dict, "paddle_dy")
    para_state_dict, opti_state_dict = paddle.framework.load( "paddle_dy")
    adam.set_state_dict(opti_state_dict)


.. py:method:: set_lr(value)

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

手动设置当前 ``optimizer`` 的学习率。

参数：
    value (float) - 需要设置的学习率的值。

返回：None


**代码示例**

.. code-block:: python

    # 这个示例需要由fleetrun启动, 用法为:
    # fleetrun --gpus=0,1 example.py
    # 脚本example.py中的代码是下面这个示例.

    import numpy as np
    import paddle
    from paddle.distributed import fleet

    paddle.disable_static()
    fleet.init(is_collective=True)

    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.fluid.dygraph.to_variable(value)

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


.. py:method:: get_lr()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

获取当前步骤的学习率。

返回：float，当前步骤的学习率。



**代码示例**

.. code-block:: python

    # 这个示例需要由fleetrun启动, 用法为:
    # fleetrun --gpus=0,1 example.py
    # 脚本example.py中的代码是下面这个示例.

    import numpy as np
    import paddle
    from paddle.distributed import fleet

    paddle.disable_static()
    fleet.init(is_collective=True)

    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.fluid.dygraph.to_variable(value)

    layer = paddle.nn.Linear(13, 5)
    adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)

    lr = adam.get_lr()
    print(lr) # 0.01


.. py:method:: step()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

执行一次优化器并进行参数更新。

返回：None。


**代码示例**

.. code-block:: python

    # 这个示例需要由fleetrun启动, 用法为:
    # fleetrun --gpus=0,1 example.py
    # 脚本example.py中的代码是下面这个示例.

    import paddle
    import paddle.nn as nn
    from paddle.distributed import fleet

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear1 = nn.Linear(10, 10)
            self._linear2 = nn.Linear(10, 1)

        def forward(self, x):
            return self._linear2(self._linear1(x))

    # 1. enable dynamic mode
    paddle.disable_static()

    # 2. initialize fleet environment
    fleet.init(is_collective=True)

    # 3. create layer & optimizer
    layer = LinearNet()
    loss_fn = nn.MSELoss()
    adam = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=layer.parameters())

    # 4. get data_parallel model using fleet
    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)

    # 5. run layer
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    print("loss:", loss.numpy())

    loss = dp_layer.scale_loss(loss)
    loss.backward()
    dp_layer.apply_collective_grads()

    adam.step()
    adam.clear_grad()


.. py:method:: clear_grad()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**


清除需要优化的参数的梯度。

返回：None。


**代码示例**

.. code-block:: python

    # 这个示例需要由fleetrun启动, 用法为:
    # fleetrun --gpus=0,1 example.py
    # 脚本example.py中的代码是下面这个示例.

    import paddle
    import paddle.nn as nn
    from paddle.distributed import fleet

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear1 = nn.Linear(10, 10)
            self._linear2 = nn.Linear(10, 1)

        def forward(self, x):
            return self._linear2(self._linear1(x))

    # 1. enable dynamic mode
    paddle.disable_static()

    # 2. initialize fleet environment
    fleet.init(is_collective=True)

    # 3. create layer & optimizer
    layer = LinearNet()
    loss_fn = nn.MSELoss()
    adam = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=layer.parameters())

    # 4. get data_parallel model using fleet
    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)

    # 5. run layer
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    print("loss:", loss.numpy())

    loss = dp_layer.scale_loss(loss)
    loss.backward()
    dp_layer.apply_collective_grads()

    adam.step()
    adam.clear_grad()


.. py:method:: minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None)


.. py:attribute:: util


