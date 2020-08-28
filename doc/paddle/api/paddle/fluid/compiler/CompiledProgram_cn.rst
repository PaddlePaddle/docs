.. _cn_api_fluid_CompiledProgram:

CompiledProgram
-------------------------------


.. py:class:: paddle.fluid.CompiledProgram(program_or_graph, build_strategy=None)

:api_attr: 声明式编程模式（静态图)



CompiledProgram根据 `build_strategy` 的配置将输入的Program或Graph进行转换和优化，例如：计算图中算子融合、计算图执行过程中开启内存/显存优化等，关于build_strategy更多信息。请参阅  ``fluid.BuildStrategy`` 。

参数：
  - **program_or_graph** (Graph|Program): 该参数为被执行的Program或Graph。
  - **build_strategy** (BuildStrategy): 通过配置build_strategy，对计算图进行转换和优化，例如：计算图中算子融合、计算图执行过程中开启内存/显存优化等。关于build_strategy更多信息，请参阅  ``fluid.BuildStrategy`` 。 默认为None。

返回：初始化后的 ``CompiledProgram`` 对象

返回类型：CompiledProgram

**代码示例**

.. code-block:: python
        
    import paddle.fluid as fluid
    import numpy

    place = fluid.CUDAPlace(0) # fluid.CPUPlace()
    exe = fluid.Executor(place)

    data = fluid.data(name='X', shape=[None, 1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = fluid.layers.mean(hidden)
    fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

    exe.run(fluid.default_startup_program())
    compiled_prog = fluid.CompiledProgram(
              fluid.default_main_program())

    x = numpy.random.random(size=(10, 1)).astype('float32')
    loss_data, = exe.run(compiled_prog,
                          feed={"X": x},
                          fetch_list=[loss.name])


.. py:method:: with_data_parallel(loss_name=None, build_strategy=None, exec_strategy=None, share_vars_from=None, places=None)

该接口用于将输入的Program或Graph进行转换，以便通过数据并行模式运行该模型。用户可以通过 `build_strategy` 和 `exec_strategy` 设置计算图构建和计算图执行过程中可以进行的一些优化，例如：将梯度聚合的AllReduce操作进行融合、指定计算图运行过程中使用的线程池大小等。**注意：如果在构建CompiledProgram和调用with_data_parallel时都指定了build_strategy，在CompiledProgram中的build_strategy会被复写，因此，如果是数据并行训练，建议在调用with_data_parallel接口时设置build_strategy**。
     
参数：
  - **loss_name** （str） - 该参数为模型最后得到的损失变量的名字，**注意：如果是模型训练，必须设置loss_name，否则计算结果可能会有问题。** 默认为：None。
  - **build_strategy** （BuildStrategy）: 通过配置build_strategy，对计算图进行转换和优化，例如：计算图中算子融合、计算图执行过程中开启内存/显存优化等。关于build_strategy更多的信息，请参阅  ``fluid.BuildStrategy`` 。 默认为：None。
  - **exec_strategy** （ExecutionStrategy） -  通过exec_strategy指定执行计算图过程可以调整的选项，例如线程池大小等。 关于exec_strategy更多信息，请参阅 ``fluid.ExecutionStrategy`` 。 默认为：None。
  - **share_vars_from** （CompiledProgram） - 如果设置了share_vars_from，当前的CompiledProgram将与share_vars_from指定的CompiledProgram共享参数值。需要设置该参数的情况：模型训练过程中需要进行模型测试，并且训练和测试都是采用数据并行模式，那么测试对应的CompiledProgram在调用with_data_parallel时，需要将share_vars_from设置为训练对应的CompiledProgram。由于CompiledProgram只有在第一次执行时才会将变量分发到其他设备上，因此share_vars_from指定的CompiledProgram必须在当前CompiledProgram之前运行。默认为：None。
  - **places** （list(CUDAPlace)|list(CPUPlace)） - 该参数指定模型运行所在的设备。如果希望在GPU0和GPU1上运行，places为[fluid.CUDAPlace(0), fluid.CUDAPlace(1)]；如果希望使用2个CPU运行，places为[fluid.CPUPlace()] * 2。 如果没有设置该参数，即该参数为None，模型执行时，将从环境变量中获取可用的设备：如果使用GPU，模型执行时，从环境变量FLAGS_selected_gpus或CUDA_VISIBLE_DEVICES中获取当前可用的设备ID；如果使用CPU，模型执行时，从环境变量CPU_NUM中获取当前可利用的CPU个数。例如：export CPU_NUM=4，如果没有设置该环境变量，执行器会在环境变量中添加该变量，并将其值设为1。默认为：None。

返回：配置之后的 ``CompiledProgram`` 对象

返回类型：CompiledProgram

.. note::
     1. 如果只是进行多卡测试，不需要设置loss_name以及share_vars_from。
     2. 如果程序中既有模型训练又有模型测试，则构建模型测试所对应的CompiledProgram时必须设置share_vars_from，否则模型测试和模型训练所使用的参数是不一致。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy
    import os

    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    parallel_places = [fluid.CUDAPlace(0), fluid.CUDAPlace(1)] if use_cuda else [fluid.CPUPlace()] * 2

    # 注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
    # 否则fluid会把逻辑核的所有数目设为CPU_NUM，
    # 在这种情况下，输入的batch size应大于CPU_NUM，
    # 否则程序会异常中断。
    if not use_cuda:
        os.environ['CPU_NUM'] = str(2)

    exe = fluid.Executor(place)

    data = fluid.data(name='X', shape=[None, 1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = fluid.layers.mean(hidden)

    test_program = fluid.default_main_program().clone(for_test=True)
    fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

    exe.run(fluid.default_startup_program())
    compiled_train_prog = fluid.CompiledProgram(
        fluid.default_main_program()).with_data_parallel(
                loss_name=loss.name, places=parallel_places)
    # 注意：如果此处不设置share_vars_from=compiled_train_prog，
    # 测试过程中用的参数与训练使用的参数是不一致
    compiled_test_prog = fluid.CompiledProgram(
        test_program).with_data_parallel(
                share_vars_from=compiled_train_prog,
                places=parallel_places)

    train_data = numpy.random.random(size=(10, 1)).astype('float32')
    loss_data, = exe.run(compiled_train_prog,
                      feed={"X": train_data},
                      fetch_list=[loss.name])
    test_data = numpy.random.random(size=(10, 1)).astype('float32')
    loss_data, = exe.run(compiled_test_prog,
                      feed={"X": test_data},
                      fetch_list=[loss.name])