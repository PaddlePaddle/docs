.. _cn_api_fluid_executor_Executor:

Executor
-------------------------------


.. py:class:: paddle.fluid.executor.Executor (place)

Executor支持单GPU、多GPU以及CPU运行。在Executor构造时，需要指定设备，在Executor执行时，需要指定被执行的Program或CompiledProgram。需要注意的是，执行器会执行Program或CompiledProgram中的所有算子。同时，需要传入运行该模型用到的scope，如果没有指定scope，执行器将使用全局scope，即fluid.global_scope()。


**示例代码**

.. code-block:: python
    
    import paddle.fluid as fluid
    import paddle.fluid.compiler as compiler
    import numpy
    import os

    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

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
    if not use_cuda:
        os.environ['CPU_NUM'] = str(2)

    compiled_prog = compiler.CompiledProgram(
        train_program).with_data_parallel(
        loss_name=loss.name)
    loss_data, = exe.run(compiled_prog,
                         feed={"X": x},
                         fetch_list=[loss.name])


参数:
    - **place** (fluid.CPUPlace|fluid.CUDAPlace(n)) – 该参数表示Executor执行所在的设备。



.. py:method:: close()


关闭执行器。

该接口主要用于对于分布式训练, 调用这个接口后不可以再使用改执行器。该接口会释放在PServers上和目前Trainer有关联的资源。

**示例代码**

.. code-block:: python
    
    import paddle.fluid as fluid

    cpu = fluid.CPUPlace()
    exe = fluid.Executor(cpu)
    # 执行训练或测试过程
    exe.close()


.. py:method:: run(program=None, feed=None, fetch_list=None, feed_var_name='feed', fetch_var_name='fetch', scope=None, return_numpy=True,use_program_cache=False)

该接口用于执行指定的Program或者CompiledProgram，**注意：执行器会执行program中的所有算子**。

**示例代码**

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

参数：  
  - **program** (Program|CompiledProgram) – 该参数为被执行的Program或CompiledProgram，如果未设置该参数，默认使用fluid.default_main_program()。默认为：None。
  - **feed** (list|dict|None) – 该变量表示模型的输入变量。如果是单卡训练，``feed`` 为 ``dict`` 类型，如果是多卡训练，参数 ``feed`` 可以是 ``dict`` 或者 ``list`` 类型变量，如果该参数类型为 ``dict`` ，feed中的数据将会被分割(split)并分送给多个设备（CPU/GPU），即输入数据被均匀分配到不同设备上；如果该参数类型为 ``list`` ，则列表中的各个元素都会直接分别被拷贝到各设备中。默认为：None。
  - **fetch_list** (list) – 该变量表示模型运行之后需要返回的变量。默认为：None。
  - **feed_var_name** (str) – 前向算子(feed operator)变量的名称。默认为："feed"。
  - **fetch_var_name** (str) – 结果获取算子(fetch operator)的输出变量名称。默认为："fetch"。
  - **scope** (Scope) – 执行这个program的域，用户可以指定不同的域。默认值：fluid.global_scope()。
  - **return_numpy** (bool) – 如果为True，则将需要返回的计算结果（fetch list中指定的变量）转化为numpy。默认为：True。
  - **use_program_cache** (bool) – 是否跨批使用缓存程序设置。设置为True时，只有当（1）程序没有用数据并行编译，并且（2）program、 feed变量名和fetch_list变量名与上一步相比没有更改时，运行速度才会更快。默认为：False。
  
返回: 根据fetch_list来获取结果

返回类型: list(numpy.array)

.. note::
     1. 如果是多卡训练，并且feed参数为dict类型，输入数据将被均匀分配到不同的卡上，例如：使用2块GPU训练，输入样本数为3，即[0, 1, 2]，经过拆分之后，GPU0上的样本数为1，即[0]，GPU1上的样本数为2，即[1, 2]。如果样本数少于设备数，程序会报错。应额外注意核对数据集的最后一个batch是否比可用的CPU核数或GPU卡数大。
     2. 如果可用的CPU核数或GPU卡数大于1，则fetch出来的结果为不同设备上的相同变量变量（fetch_list中的变量）拼接在一起。


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

返回: None

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

返回: None

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


