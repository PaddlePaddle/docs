#################
 fluid.executor
#################



.. _cn_api_fluid_executor_Executor:

Executor
-------------------------------


.. py:class:: paddle.fluid.executor.Executor (place)




执行引擎（Executor）使用python脚本驱动，支持在单/多GPU、单/多CPU环境下运行。
Python Executor可以接收传入的program,并根据feed map(输入映射表)和fetch_list(结果获取表)
向program中添加feed operators(数据输入算子)和fetch operators（结果获取算子)。
feed map为该program提供输入数据。fetch_list提供program训练结束后用户预期的变量（或识别类场景中的命名）。

应注意，执行器会执行program中的所有算子而不仅仅是依赖于fetch_list的那部分。

Executor将全局变量存储到全局作用域中，并为临时变量创建局部作用域。
当每一mini-batch上的前向/反向运算完成后，局部作用域的内容将被废弃，
但全局作用域中的变量将在Executor的不同执行过程中一直存在。


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
    - **place** (fluid.CPUPlace|fluid.CUDAPlace(n)) – 指明了 ``Executor`` 的执行场所



.. py:method:: close()


关闭这个执行器(Executor)。

调用这个方法后不可以再使用这个执行器。 对于分布式训练, 该函数会释放在PServers上和目前Trainer有关联的资源。

**示例代码**

.. code-block:: python
    
    import paddle.fluid as fluid

    cpu = fluid.CPUPlace()
    exe = fluid.Executor(cpu)
    # 执行训练或测试过程
    exe.close()


.. py:method:: run(program=None, feed=None, fetch_list=None, feed_var_name='feed', fetch_var_name='fetch', scope=None, return_numpy=True,use_program_cache=False)


调用该执行器对象的此方法可以执行program。通过feed map提供待学习数据，以及借助fetch_list得到相应的结果。
Python执行器(Executor)可以接收传入的program,并根据输入映射表(feed map)和结果获取表(fetch_list)
向program中添加数据输入算子(feed operators)和结果获取算子（fetch operators)。
feed map为该program提供输入数据。fetch_list提供program训练结束后用户预期的变量（或识别类场景中的命名）。

应注意，执行器会执行program中的所有算子而不仅仅是依赖于fetch_list的那部分。

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
  - **program** (Program|CompiledProgram) – 需要执行的program,如果没有给定那么默认使用default_main_program (未编译的)
  - **feed** (dict) – 前向输入的变量，数据,词典dict类型, 例如 {“image”: ImageData, “label”: LabelData}
  - **fetch_list** (list) – 用户想得到的变量或者命名的列表, 该方法会根据这个列表给出结果
  - **feed_var_name** (str) – 前向算子(feed operator)变量的名称
  - **fetch_var_name** (str) – 结果获取算子(fetch operator)的输出变量名称
  - **scope** (Scope) – 执行这个program的域，用户可以指定不同的域。缺省为全局域
  - **return_numpy** (bool) – 如果为True,则将结果张量（fetched tensor）转化为numpy
  - **use_program_cache** (bool) – 是否跨批使用缓存程序设置。设置为True时，只有当（1）程序没有用数据并行编译，并且（2）program、 feed变量名和fetch_list变量名与上一步相比没有更改时，运行速度才会更快。
  
返回: 根据fetch_list来获取结果

返回类型: list(numpy.array)


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
        exe.infer_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset)


.. _cn_api_fluid_executor_global_scope:

global_scope
-------------------------------

.. py:function:: paddle.fluid.global_scope()


获取全局/默认作用域实例。很多api使用默认 ``global_scope`` ，例如 ``Executor.run`` 。

**示例代码**

.. code-block:: python

        import paddle.fluid as fluid
        import numpy

        fluid.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), fluid.CPUPlace())
        numpy.array(fluid.global_scope().find_var("data").get_tensor())

返回：全局/默认作用域实例

返回类型：Scope






.. _cn_api_fluid_executor_scope_guard:

scope_guard
-------------------------------

.. py:function:: paddle.fluid.executor.scope_guard (scope)


修改全局/默认作用域（scope）,  运行时中的所有变量都将分配给新的scope。

参数：
    - **scope** - 新的全局/默认 scope。

**代码示例**

.. code-block:: python

    import numpy

    new_scope = fluid.Scope()
    with fluid.scope_guard(new_scope):
         fluid.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), fluid.CPUPlace())
    numpy.array(new_scope.find_var("data").get_tensor())













