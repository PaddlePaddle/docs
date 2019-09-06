.. _cn_api_fluid_BuildStrategy:

BuildStrategy
-------------------------------

.. py:class::  paddle.fluid.BuildStrategy

``BuildStrategy`` 使用户更精准地控制 ``ParallelExecutor`` 中SSA图的建造方法。可通过设置 ``ParallelExecutor`` 中的 ``BuildStrategy`` 成员来实现此功能。

**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce


.. py:attribute:: debug_graphviz_path

str类型。它表明了以graphviz格式向文件中写入SSA图的路径，有利于调试。 默认值为""。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.debug_graphviz_path = "./graph"


.. py:attribute:: enable_sequential_execution

类型是BOOL。 如果设置为True，则ops的执行顺序将与program中的执行顺序相同。 默认为False。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_sequential_execution = True


.. py:attribute:: fuse_broadcast_ops
     
bool类型。它表明了是否融合（fuse）broadcast ops。值得注意的是，在Reduce模式中，融合broadcast ops可以使程序运行更快，因为这个过程等同于延迟执行所有的broadcast ops。在这种情况下，所有的nccl streams仅用于一段时间内的NCCLReduce操作。默认为False。
     
.. py:attribute:: fuse_elewise_add_act_ops

bool类型。它表明了是否融合（fuse）elementwise_add_op和activation_op。这会使整体执行过程更快一些。默认为False。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.fuse_elewise_add_act_ops = True


.. py:attribute:: fuse_relu_depthwise_conv

BOOL类型，fuse_relu_depthwise_conv指示是否融合relu和depthwise_conv2d，它会节省GPU内存并可能加速执行过程。 此选项仅适用于GPU设备。 默认为False。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.fuse_relu_depthwise_conv = True

.. py:attribute:: gradient_scale_strategy

``fluid.BuildStrategy.GradientScaleStrategy`` 类型。在 ``ParallelExecutor`` 中，存在三种定义 *loss@grad* 的方式，分别为 ``CoeffNumDevice``, ``One`` 与 ``Customized``。默认情况下， ``ParallelExecutor`` 根据设备数目来设置 *loss@grad* 。如果你想自定义 *loss@grad* ，你可以选择 ``Customized`` 方法。默认为 ``CoeffNumDevice`` 。

**代码示例**

.. code-block:: python

    import paddle.fluid.compiler as compiler
    import numpy
    import os

    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # NOTE: 如果你使用CPU计算，需要指定CPU_NUM, 否则,fluid
    # 将使用所有的核的数目作为CPU_NUM,
    # 这种情况下，输入的batch size应该大于CPU_NUM, 否则, 
    # 进程将会因为异常而失败。
    if not use_cuda:
        os.environ['CPU_NUM'] = str(2)
        places = fluid.cpu_places()
    else:
        places = places = fluid.cuda_places()

    data = fluid.layers.data(name='X', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = fluid.layers.mean(hidden)
    fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

    fluid.default_startup_program().random_seed=1
    exe.run(fluid.default_startup_program())

    build_strategy = fluid.BuildStrategy()
    build_strategy.gradient_scale_strategy = \
         fluid.BuildStrategy.GradientScaleStrategy.Customized
    compiled_prog = compiler.CompiledProgram(
         fluid.default_main_program()).with_data_parallel(
                  loss_name=loss.name, build_strategy=build_strategy,
                  places = places)

    dev_count =  len(places)
    x = numpy.random.random(size=(10, 1)).astype('float32')
    loss_grad = numpy.ones((dev_count)).astype("float32") * 0.01
    loss_grad_name = loss.name+"@GRAD"
    loss_data = exe.run(compiled_prog,
                     feed={"X": x, loss_grad_name : loss_grad},
                     fetch_list=[loss.name, loss_grad_name])

.. py:attribute:: memory_optimize

bool类型或者None。设为True时可用于减少总内存消耗。
默认为None。None表示框架会自动选择使用或者不使用优化策略。当前，None意味着当GC不能使用时，优化策略将被使用。
True表示可使用，False表示不可使用。默认为None。

.. py:attribute:: reduce_strategy

``fluid.BuildStrategy.ReduceStrategy`` 类型。在 ``ParallelExecutor`` 中，存在两种减少策略（reduce strategy），即 ``AllReduce`` 和 ``Reduce`` 。如果你需要在所有执行设备上独立地进行参数优化，可以使用 ``AllReduce`` 。反之，如果使用 ``Reduce`` 策略，所有参数的优化将均匀地分配给不同的执行设备，随之将优化后的参数广播给其他执行设备。
默认值为 ``AllReduce`` 。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

.. py:attribute:: remove_unnecessary_lock

BOOL类型。如果设置为True, GPU操作中的一些锁将被释放， ``ParallelExecutor`` 将运行得更快，默认为 True。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.remove_unnecessary_lock = True


.. py:attribute:: sync_batch_norm

类型为bool， ``sync_batch_norm`` 表示是否使用同步的批正则化，即在训练阶段通过多个设备同步均值和方差。

当前的实现不支持FP16训练和CPU。仅在一台机器上进行同步式批正则，不适用于多台机器。

默认为 False。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.sync_batch_norm = True


