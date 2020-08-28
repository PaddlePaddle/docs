.. _cn_api_fluid_executor:

Executor
-------------------------------



.. py:class:: paddle.fluid.Executor (place=None)

:api_attr: 声明式编程模式（静态图)



Executor支持单GPU、多GPU以及CPU运行。

参数：
    - **place** (fluid.CPUPlace()|fluid.CUDAPlace(N)|None) – 该参数表示Executor执行所在的设备，这里的N为GPU对应的ID。当该参数为 `None` 时，PaddlePaddle会根据其安装版本设置默认的运行设备。当安装的Paddle为CPU版时，默认运行设置会设置成 `CPUPlace()` ，而当Paddle为GPU版时，默认运行设备会设置成 `CUDAPlace(0)` 。默认值为None。
  
返回：初始化后的 ``Executor`` 对象

返回类型：Executor

**示例代码**

.. code-block:: python
    
    import paddle.fluid as fluid
    import paddle.fluid.compiler as compiler
    import numpy
    import os

    # 显式设置运行设备
    # use_cuda = True
    # place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    # exe = fluid.Executor(place)

    # 如果不显示设置运行设备，PaddlePaddle会设置默认运行设备
    exe = fluid.Executor()

    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

    # 仅运行一次startup program
    # 不需要优化/编译这个startup program
    startup_program.random_seed=1
    exe.run(startup_program)

    # 无需编译，直接运行main program
    x = numpy.random.random(size=(10, 1)).astype('float32')
    loss_data, = exe.run(train_program,
                     feed={"X": x},
                     fetch_list=[loss.name])

    # 另一种方法是，编译这个main program然后运行。
    # 参考CompiledProgram以获取更多信息。
    # 注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
    # 否则fluid会把逻辑核的所有数目设为CPU_NUM，
    # 在这种情况下，输入的batch size应大于CPU_NUM，
    # 否则程序会异常中断。

    # 显式设置运行设备
    # if not use_cuda:
    #    os.environ['CPU_NUM'] = str(2)

    # 未显示设置运行设备且安装的Paddle为CPU版本
    os.environ['CPU_NUM'] = str(2)

    compiled_prog = compiler.CompiledProgram(
        train_program).with_data_parallel(
        loss_name=loss.name)
    loss_data, = exe.run(compiled_prog,
                         feed={"X": x},
                         fetch_list=[loss.name])

.. py:method:: close()


关闭执行器。该接口主要用于对于分布式训练，调用该接口后不可以再使用该执行器。该接口会释放在PServers上和目前Trainer有关联的资源。

返回：无

**示例代码**

.. code-block:: python
    
    import paddle.fluid as fluid

    cpu = fluid.CPUPlace()
    exe = fluid.Executor(cpu)
    # 执行训练或测试过程
    exe.close()


.. py:method:: run(program=None, feed=None, fetch_list=None, feed_var_name='feed', fetch_var_name='fetch', scope=None, return_numpy=True, use_program_cache=False, return_merged=True)

执行指定的Program或者CompiledProgram。需要注意的是，执行器会执行Program或CompiledProgram中的所有算子，而不会根据fetch_list对Program或CompiledProgram中的算子进行裁剪。同时，需要传入运行该模型用到的scope，如果没有指定scope，执行器将使用全局scope，即fluid.global_scope()。

参数：  
  - **program** (Program|CompiledProgram) – 该参数为被执行的Program或CompiledProgram，如果未提供该参数，即该参数为None，在该接口内，main_program将被设置为fluid.default_main_program()。默认为：None。
  - **feed** (list|dict) – 该参数表示模型的输入变量。如果是单卡训练，``feed`` 为 ``dict`` 类型，如果是多卡训练，参数 ``feed`` 可以是 ``dict`` 或者 ``list`` 类型变量，如果该参数类型为 ``dict`` ，feed中的数据将会被分割(split)并分送给多个设备（CPU/GPU），即输入数据被均匀分配到不同设备上；如果该参数类型为 ``list`` ，则列表中的各个元素都会直接分别被拷贝到各设备中。默认为：None。
  - **fetch_list** (list) – 该参数表示模型运行之后需要返回的变量。默认为：None。
  - **feed_var_name** (str) – 该参数表示数据输入算子(feed operator)的输入变量名称。默认为："feed"。
  - **fetch_var_name** (str) – 该参数表示结果获取算子(fetch operator)的输出变量名称。默认为："fetch"。
  - **scope** (Scope) – 该参数表示执行当前program所使用的作用域，用户可以为不同的program指定不同的作用域。默认值：fluid.global_scope()。
  - **return_numpy** (bool) – 该参数表示是否将返回的计算结果（fetch list中指定的变量）转化为numpy；如果为False，则每个变量返回的类型为LoDTensor，否则返回变量的类型为numpy.ndarray。默认为：True。
  - **use_program_cache** (bool) – 该参数表示是否对输入的Program进行缓存。如果该参数为True，在以下情况时，模型运行速度可能会更快：输入的program为 ``fluid.Program`` ，并且模型运行过程中，调用该接口的参数（program、 feed变量名和fetch_list变量）名始终不变。默认为：False。
  - **return_merged** (bool) – 该参数表示是否按照执行设备维度将返回的计算结果（fetch list中指定的变量）进行合并。如果 ``return_merged`` 设为False，返回值类型是一个Tensor的二维列表（ ``return_numpy`` 设为Fasle时）或者一个numpy.ndarray的二维列表（ ``return_numpy`` 设为True时）。如果 ``return_merged`` 设为True，返回值类型是一个Tensor的一维列表（ ``return_numpy`` 设为Fasle时）或者一个numpy.ndarray的一维列表（ ``return_numpy`` 设为True时）。更多细节请参考示例代码2。如果返回的计算结果是变长的，请设置 ``return_merged`` 为False，即不按照执行设备维度合并返回的计算结果。该参数的默认值为True，但这仅是为了兼容性考虑，在未来的版本中默认值可能会更改为False。

返回：返回fetch_list中指定的变量值

返回类型：List

.. note::
     1. 如果是多卡训练，并且feed参数为dict类型，输入数据将被均匀分配到不同的卡上，例如：使用2块GPU训练，输入样本数为3，即[0, 1, 2]，经过拆分之后，GPU0上的样本数为1，即[0]，GPU1上的样本数为2，即[1, 2]。如果样本数少于设备数，程序会报错，因此运行模型时，应额外注意数据集的最后一个batch的样本数是否少于当前可用的CPU核数或GPU卡数，如果是少于，建议丢弃该batch。
     2. 如果可用的CPU核数或GPU卡数大于1，则fetch出来的结果为不同设备上的相同变量值（fetch_list中的变量）在第0维拼接在一起。


**示例代码1**

.. code-block:: python

            import paddle.fluid as fluid
            import numpy
     
            #首先创建执行引擎
            place = fluid.CPUPlace() # fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
     
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            adam = fluid.optimizer.Adam()
            adam.minimize(loss)
     
            #仅运行startup程序一次
            exe.run(fluid.default_startup_program())

            x = numpy.random.random(size=(10, 1)).astype('float32')
            outs = exe.run(feed={'X': x},
                           fetch_list=[loss.name])


**示例代码2**

.. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            # 创建Executor对象
            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            data = fluid.data(name='X', shape=[None, 1], dtype='float32')
            class_dim = 2
            prediction = fluid.layers.fc(input=data, size=class_dim)
            loss = fluid.layers.mean(prediction)
            adam = fluid.optimizer.Adam()
            adam.minimize(loss)
            # 运行且仅运行一次startup program
            exe.run(fluid.default_startup_program())
            build_strategy = fluid.BuildStrategy()
            binary = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(
                loss_name=loss.name, build_strategy=build_strategy)
            batch_size = 6
            x = np.random.random(size=(batch_size, 1)).astype('float32')
            # 1) 设置 return_merged 参数为False以获取不合并的计算结果：
            unmerged_prediction, = exe.run(binary, feed={'X': x},
                fetch_list=[prediction.name],
                return_merged=False)
            # 如果用户使用两个GPU卡来运行此python代码示例，输出结果将为(2, 3, class_dim)。
            # 输出结果中第一个维度值代表所使用的GPU卡数，而第二个维度值代表batch_size和所使用
            # 的GPU卡数之商。
            print("The unmerged prediction shape: {}".format(np.array(unmerged_prediction).shape))
            print(unmerged_prediction)
            # 2) 设置 return_merged 参数为True以获取合并的计算结果：
            merged_prediction, = exe.run(binary, feed={'X': x},
                fetch_list=[prediction.name],
                return_merged=True)
            # 如果用户使用两个GPU卡来运行此python代码示例，输出结果将为(6, class_dim)。输出结果
            # 中第一个维度值代表batch_size值。
            print("The merged prediction shape: {}".format(np.array(merged_prediction).shape))
            print(merged_prediction)
            # 输出:
            # The unmerged prediction shape: (2, 3, 2)
            # [array([[-0.37620035, -0.19752218],
            #        [-0.3561043 , -0.18697084],
            #        [-0.24129935, -0.12669306]], dtype=float32), array([[-0.24489994, -0.12858354],
            #        [-0.49041364, -0.25748932],
            #        [-0.44331917, -0.23276259]], dtype=float32)]
            # The merged prediction shape: (6, 2)
            # [[-0.37789783 -0.19921964]
            #  [-0.3577645  -0.18863106]
            #  [-0.24274671 -0.12814042]
            #  [-0.24635398 -0.13003758]
            #  [-0.49232286 -0.25939852]
            #  [-0.44514108 -0.2345845 ]]


.. py:method:: infer_from_dataset(program=None, dataset=None, scope=None, thread=0, debug=False, fetch_list=None, fetch_info=None, print_period=100)

infer_from_dataset的文档与train_from_dataset几乎完全相同，只是在分布式训练中，推进梯度将在infer_from_dataset中禁用。 infer_from_dataset（）可以非常容易地用于多线程中的评估。

参数：  
  - **program** (Program|CompiledProgram) – 需要执行的program,如果没有给定那么默认使用default_main_program (未编译的)
  - **dataset** (paddle.fluid.Dataset) – 在此函数外创建的数据集，用户应当在调用函数前提供完整定义的数据集。必要时请检查Dataset文件。默认为None
  - **scope** (Scope) – 执行这个program的域，用户可以指定不同的域。默认为全局域
  - **thread** (int) – 用户想要在这个函数中运行的线程数量。线程的实际数量为min(Dataset.thread_num, thread)，如果thread > 0，默认为0
  - **debug** (bool) – 是否开启debug模式，默认为False
  - **fetch_list** (Variable List) – 返回变量列表，每个变量都会在训练过程中被打印出来，默认为None
  - **fetch_info** (String List) – 每个变量的打印信息，默认为None
  - **print_period** (int) – 每两次打印之间间隔的mini-batches的数量，默认为100

返回：None

**示例代码**

.. code-block:: python

  import paddle.fluid as fluid
  place = fluid.CPUPlace() # 使用GPU时可设置place = fluid.CUDAPlace(0)
  exe = fluid.Executor(place)
  x = fluid.layers.data(name="x", shape=[10, 10], dtype="int64")
  y = fluid.layers.data(name="y", shape=[1], dtype="int64", lod_level=1)
  dataset = fluid.DatasetFactory().create_dataset()
  dataset.set_use_var([x, y])
  dataset.set_thread(1)
  filelist = [] # 您可以设置您自己的filelist，如filelist = ["dataA.txt"]
  dataset.set_filelist(filelist)
  exe.run(fluid.default_startup_program())
  exe.infer_from_dataset(program=fluid.default_main_program(),dataset=dataset)
     

.. py:method:: train_from_dataset(program=None, dataset=None, scope=None, thread=0, debug=False, fetch_list=None, fetch_info=None, print_period=100)

从预定义的数据集中训练。 数据集在paddle.fluid.dataset中定义。 给定程序（或编译程序），train_from_dataset将使用数据集中的所有数据样本。 输入范围可由用户给出。 默认情况下，范围是global_scope()。训练中的线程总数是thread。 训练中使用的线程数将是数据集中threadnum的最小值，同时也是此接口中线程的值。 可以设置debug，以便执行器显示所有算子的运行时间和当前训练任务的吞吐量。

注意：train_from_dataset将销毁每次运行在executor中创建的所有资源。

参数：  
  - **program** (Program|CompiledProgram) – 需要执行的program,如果没有给定那么默认使用default_main_program (未编译的)
  - **dataset** (paddle.fluid.Dataset) – 在此函数外创建的数据集，用户应当在调用函数前提供完整定义的数据集。必要时请检查Dataset文件。默认为None
  - **scope** (Scope) – 执行这个program的域，用户可以指定不同的域。默认为全局域
  - **thread** (int) – 用户想要在这个函数中运行的线程数量。线程的实际数量为min(Dataset.thread_num, thread)，如果thread > 0，默认为0
  - **debug** (bool) – 是否开启debug模式，默认为False
  - **fetch_list** (Variable List) – 返回变量列表，每个变量都会在训练过程中被打印出来，默认为None
  - **fetch_info** (String List) – 每个变量的打印信息，默认为None
  - **print_period** (int) – 每两次打印之间间隔的mini-batches的数量，默认为100

返回：None

**示例代码**

.. code-block:: python

        import paddle.fluid as fluid

        place = fluid.CPUPlace() # 通过设置place = fluid.CUDAPlace(0)使用GPU
        exe = fluid.Executor(place)
        x = fluid.layers.data(name="x", shape=[10, 10], dtype="int64")
        y = fluid.layers.data(name="y", shape=[1], dtype="int64", lod_level=1)
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var([x, y])
        dataset.set_thread(1)
        filelist = [] # 您可以设置您自己的filelist，如filelist = ["dataA.txt"]
        dataset.set_filelist(filelist)
        exe.run(fluid.default_startup_program())
        exe.train_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset)
