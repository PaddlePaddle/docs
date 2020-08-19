.. _cn_api_fluid_dygraph_DataParallel:

DataParallel
------------

.. py:class:: paddle.fluid.dygraph.DataParallel(layers, strategy)

:api_attr: 命令式编程模式（动态图)

通过数据并行模式执行动态图模型。

目前，``DataParallel`` 仅支持以多进程的方式执行动态图模型。使用方式如下：

``python -m paddle.distributed.launch –selected_gpus=0,1 dynamic_graph_test.py``

其中 ``dynamic_graph_test.py`` 脚本的代码可以是下面的示例代码。

参数：
    - **Layer** (Layer) - 需要通过数据并行方式执行的模型。
    - **strategy** (ParallelStrategy) - 数据并行的策略，包括并行执行的环境配置。

返回：支持数据并行的 ``Layer``

返回类型：Layer实例

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid

    place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
    with fluid.dygraph.guard(place):

        # prepare the data parallel context
        strategy = fluid.dygraph.prepare_context()

        linear = fluid.dygraph.Linear(1, 10, act="softmax")
        adam = fluid.optimizer.AdamOptimizer(
            learning_rate=0.001, parameter_list=linear.parameters())

        # make the module become the data parallelism module
        linear = fluid.dygraph.DataParallel(linear, strategy)

        x_data = np.random.random(size=[10, 1]).astype(np.float32)
        data = fluid.dygraph.to_variable(x_data)

        hidden = linear(data)
        avg_loss = fluid.layers.mean(hidden)

        # scale the loss according to the number of trainers.
        avg_loss = linear.scale_loss(avg_loss)

        avg_loss.backward()

        # collect the gradients of trainers.
        linear.apply_collective_grads()

        adam.minimize(avg_loss)
        linear.clear_gradients()

.. py:method:: scale_loss(loss)

缩放模型损失值 ``loss`` 。在数据并行模式中，损失值 ``loss`` 需要根据并行训练进程的数目进行缩放。

如果不在数据并行模式下，会直接返回原 ``loss`` 。

参数：
    - **loss** (Variable) - 当前模型的损失值。

返回：缩放后的损失值 ``loss``

返回类型：Variable

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid

    place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
    with fluid.dygraph.guard(place):

        # prepare the data parallel context
        strategy = fluid.dygraph.prepare_context()

        linear = fluid.dygraph.Linear(1, 10, act="softmax")
        adam = fluid.optimizer.AdamOptimizer(
            learning_rate=0.001, parameter_list=linear.parameters())

        # make the module become the data parallelism module
        linear = fluid.dygraph.DataParallel(linear, strategy)

        x_data = np.random.random(size=[10, 1]).astype(np.float32)
        data = fluid.dygraph.to_variable(x_data)

        hidden = linear(data)
        avg_loss = fluid.layers.mean(hidden)

        # scale the loss according to the number of trainers.
        avg_loss = linear.scale_loss(avg_loss)

        avg_loss.backward()

        # collect the gradients of trainers.
        linear.apply_collective_grads()

        adam.minimize(avg_loss)
        linear.clear_gradients()


.. py:method:: apply_collective_grads()

AllReduce（规约）参数的梯度值。

返回：无

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid

    place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
    with fluid.dygraph.guard(place):

        # prepare the data parallel context
        strategy = fluid.dygraph.prepare_context()

        linear = fluid.dygraph.Linear(1, 10, act="softmax")
        adam = fluid.optimizer.AdamOptimizer(
            learning_rate=0.001, parameter_list=linear.parameters())

        # make the module become the data parallelism module
        linear = fluid.dygraph.DataParallel(linear, strategy)

        x_data = np.random.random(size=[10, 1]).astype(np.float32)
        data = fluid.dygraph.to_variable(x_data)

        hidden = linear(data)
        avg_loss = fluid.layers.mean(hidden)

        # scale the loss according to the number of trainers.
        avg_loss = linear.scale_loss(avg_loss)

        avg_loss.backward()

        # collect the gradients of trainers.
        linear.apply_collective_grads()

        adam.minimize(avg_loss)
        linear.clear_gradients()
