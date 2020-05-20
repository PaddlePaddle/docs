.. _cn_api_fluid_BuildStrategy:

BuildStrategy
-------------------------------


.. py:class:: paddle.fluid.BuildStrategy

:api_attr: 声明式编程模式（静态图)



``BuildStrategy`` 使用户更方便地控制 :ref:`cn_api_fluid_ParallelExecutor` 中计算图的建造方法，可通过设置 ``ParallelExecutor`` 中的 ``BuildStrategy`` 成员来实现此功能。

**代码示例**

.. code-block:: python

    import paddle
    import os
    import numpy as np
    import paddle.fluid as fluid
    
    os.environ['CPU_NUM'] = '2'
    places = fluid.cpu_places()
    
    data = fluid.layers.data(name='x', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = paddle.mean(hidden)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
    build_strategy = paddle.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.reduce_strategy = paddle.BuildStrategy.ReduceStrategy.Reduce
    program = fluid.compiler.CompiledProgram(paddle.default_main_program())
    program = program.with_data_parallel(loss_name=loss.name, build_strategy=
        build_strategy, places=places)

.. py:attribute:: debug_graphviz_path

str类型。表示以graphviz格式向文件中写入计算图的路径，有利于调试。默认值为空字符串。

**代码示例**

.. code-block:: python

    import paddle
    import os
    import numpy as np
    import paddle.fluid as fluid
    
    os.environ['CPU_NUM'] = '2'
    places = fluid.cpu_places()
    
    data = fluid.layers.data(name='x', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = paddle.mean(hidden)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
    build_strategy = paddle.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.reduce_strategy = paddle.BuildStrategy.ReduceStrategy.Reduce
    program = fluid.compiler.CompiledProgram(paddle.default_main_program())
    program = program.with_data_parallel(loss_name=loss.name, build_strategy=
        build_strategy, places=places)

.. py:attribute:: enable_sequential_execution

bool类型。如果设置为True，则算子的执行顺序将与算子定义的执行顺序相同。默认为False。

**代码示例**

.. code-block:: python

    import paddle
    import os
    import numpy as np
    import paddle.fluid as fluid
    
    os.environ['CPU_NUM'] = '2'
    places = fluid.cpu_places()
    
    data = fluid.layers.data(name='x', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = paddle.mean(hidden)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
    build_strategy = paddle.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.reduce_strategy = paddle.BuildStrategy.ReduceStrategy.Reduce
    program = fluid.compiler.CompiledProgram(paddle.default_main_program())
    program = program.with_data_parallel(loss_name=loss.name, build_strategy=
        build_strategy, places=places)

.. py:attribute:: fuse_broadcast_ops
     
bool类型。表明是否融合(fuse) broadcast ops。该选项指在Reduce模式下有效，使程序运行更快。默认为False。

**代码示例**

.. code-block:: python

    import paddle
    import os
    import numpy as np
    import paddle.fluid as fluid
    
    os.environ['CPU_NUM'] = '2'
    places = fluid.cpu_places()
    
    data = fluid.layers.data(name='x', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = paddle.mean(hidden)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
    build_strategy = paddle.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.reduce_strategy = paddle.BuildStrategy.ReduceStrategy.Reduce
    program = fluid.compiler.CompiledProgram(paddle.default_main_program())
    program = program.with_data_parallel(loss_name=loss.name, build_strategy=
        build_strategy, places=places)

.. py:attribute:: fuse_elewise_add_act_ops

bool类型。表明是否融合(fuse) elementwise_add_op和activation_op。这会使整体执行过程更快。默认为False。

**代码示例**

.. code-block:: python

    import paddle
    import os
    import numpy as np
    import paddle.fluid as fluid
    
    os.environ['CPU_NUM'] = '2'
    places = fluid.cpu_places()
    
    data = fluid.layers.data(name='x', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = paddle.mean(hidden)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
    build_strategy = paddle.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.reduce_strategy = paddle.BuildStrategy.ReduceStrategy.Reduce
    program = fluid.compiler.CompiledProgram(paddle.default_main_program())
    program = program.with_data_parallel(loss_name=loss.name, build_strategy=
        build_strategy, places=places)

.. py:attribute:: fuse_relu_depthwise_conv

bool类型。表明是否融合(fuse) relu和depthwise_conv2d，节省GPU内存并可能加速执行过程。此选项仅适用于GPU设备。默认为False。

**代码示例**

.. code-block:: python

    import paddle
    import os
    import numpy as np
    import paddle.fluid as fluid
    
    os.environ['CPU_NUM'] = '2'
    places = fluid.cpu_places()
    
    data = fluid.layers.data(name='x', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = paddle.mean(hidden)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
    build_strategy = paddle.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.reduce_strategy = paddle.BuildStrategy.ReduceStrategy.Reduce
    program = fluid.compiler.CompiledProgram(paddle.default_main_program())
    program = program.with_data_parallel(loss_name=loss.name, build_strategy=
        build_strategy, places=places)

.. py:attribute:: gradient_scale_strategy

``fluid.BuildStrategy.GradientScaleStrategy`` 类型。在 ``ParallelExecutor`` 中，存在三种定义loss对应梯度( *loss@grad* )的方式，分别为 ``CoeffNumDevice``, ``One`` 与 ``Customized``。默认情况下， ``ParallelExecutor`` 根据设备数目来设置 *loss@grad* 。如果用户需要自定义 *loss@grad* ，可以选择 ``Customized`` 方法。默认为 ``CoeffNumDevice`` 。

**代码示例**

.. code-block:: python

    import paddle
    import os
    import numpy as np
    import paddle.fluid as fluid
    
    os.environ['CPU_NUM'] = '2'
    places = fluid.cpu_places()
    
    data = fluid.layers.data(name='x', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = paddle.mean(hidden)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
    build_strategy = paddle.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.reduce_strategy = paddle.BuildStrategy.ReduceStrategy.Reduce
    program = fluid.compiler.CompiledProgram(paddle.default_main_program())
    program = program.with_data_parallel(loss_name=loss.name, build_strategy=
        build_strategy, places=places)

.. py:attribute:: memory_optimize

bool类型或None。设为True时可用于减少总内存消耗，False表示不使用，None表示框架会自动选择使用或者不使用优化策略。当前，None意味着当GC不能使用时，优化策略将被使用。默认为None。

.. py:attribute:: reduce_strategy

``fluid.BuildStrategy.ReduceStrategy`` 类型。在 ``ParallelExecutor`` 中，存在两种参数梯度聚合策略，即 ``AllReduce`` 和 ``Reduce`` 。如果用户需要在所有执行设备上独立地进行参数更新，可以使用 ``AllReduce`` 。如果使用 ``Reduce`` 策略，所有参数的优化将均匀地分配给不同的执行设备，随之将优化后的参数广播给其他执行设备。
默认值为 ``AllReduce`` 。

**代码示例**

.. code-block:: python

    import paddle
    import os
    import numpy as np
    import paddle.fluid as fluid
    
    os.environ['CPU_NUM'] = '2'
    places = fluid.cpu_places()
    
    data = fluid.layers.data(name='x', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = paddle.mean(hidden)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
    build_strategy = paddle.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.reduce_strategy = paddle.BuildStrategy.ReduceStrategy.Reduce
    program = fluid.compiler.CompiledProgram(paddle.default_main_program())
    program = program.with_data_parallel(loss_name=loss.name, build_strategy=
        build_strategy, places=places)

.. py:attribute:: remove_unnecessary_lock

bool类型。设置True会去除GPU操作中的一些锁操作， ``ParallelExecutor`` 将运行得更快，默认为True。

**代码示例**

.. code-block:: python

    import paddle
    import os
    import numpy as np
    import paddle.fluid as fluid
    
    os.environ['CPU_NUM'] = '2'
    places = fluid.cpu_places()
    
    data = fluid.layers.data(name='x', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = paddle.mean(hidden)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
    build_strategy = paddle.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.reduce_strategy = paddle.BuildStrategy.ReduceStrategy.Reduce
    program = fluid.compiler.CompiledProgram(paddle.default_main_program())
    program = program.with_data_parallel(loss_name=loss.name, build_strategy=
        build_strategy, places=places)

.. py:attribute:: sync_batch_norm

bool类型。表示是否使用同步的批正则化，即在训练阶段通过多个设备同步均值和方差。当前的实现不支持FP16训练和CPU。并且目前**仅支持**仅在一台机器上进行同步式批正则。默认为 False。

**代码示例**

.. code-block:: python

    import paddle
    import os
    import numpy as np
    import paddle.fluid as fluid
    
    os.environ['CPU_NUM'] = '2'
    places = fluid.cpu_places()
    
    data = fluid.layers.data(name='x', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = paddle.mean(hidden)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
    build_strategy = paddle.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.reduce_strategy = paddle.BuildStrategy.ReduceStrategy.Reduce
    program = fluid.compiler.CompiledProgram(paddle.default_main_program())
    program = program.with_data_parallel(loss_name=loss.name, build_strategy=
        build_strategy, places=places)

