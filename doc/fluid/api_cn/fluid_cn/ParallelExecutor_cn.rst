.. _cn_api_fluid_ParallelExecutor:

ParallelExecutor
-------------------------------

.. py:class:: paddle.fluid.ParallelExecutor(use_cuda, loss_name=None, main_program=None, share_vars_from=None, exec_strategy=None, build_strategy=None, num_trainers=1, trainer_id=0, scope=None)




``ParallelExecutor`` 专门设计用来实现数据并行计算，着力于向不同结点(node)分配数据，并行地在不同结点中对数据进行操作。如果在GPU上使用该类运行程序，node则用来指代GPU， ``ParallelExecutor`` 也将自动获取在当前机器上可用的GPU资源。如果在CPU上进行操作，node则指代CPU，同时你也可以通过添加环境变量 ``CPU_NUM`` 来设置CPU设备的个数。例如，``CPU_NUM=4``。但是如果没有设置该环境变量，该类会调用 ``multiprocessing.cpu_count`` 来获取当前系统中CPU的个数。

**示例代码**

.. code-block:: python

        import paddle.fluid as fluid
        import numpy
        import os
     
        use_cuda = True
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
     
        # 注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
        # 否则fluid会把逻辑核的所有数目设为CPU_NUM，
        # 在这种情况下，输入的batch size应大于CPU_NUM，
        # 否则程序会异常中断。
        if not use_cuda:
            os.environ['CPU_NUM'] = str(2)
     
        exe = fluid.Executor(place)
     
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            test_program = fluid.default_main_program().clone(for_test=True)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
     
        startup_program.random_seed=1
        exe.run(startup_program)
     
        train_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                           main_program=train_program,
                                           loss_name=loss.name)
        test_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                          main_program=test_program,
                                          share_vars_from=train_exe)
     
        x = numpy.random.random(size=(10, 1)).astype('float32')
        loss_data, = train_exe.run(feed={"X": x},
                                   fetch_list=[loss.name])
     
        loss_data, = test_exe.run(feed={"X": x},
                                  fetch_list=[loss.name])

参数:
    - **use_cuda** (bool) – 是否使用CUDA
    - **loss_name** (str) – 在训练阶段，必须提供loss function名称。默认为None
    - **main_program** (Program) – 需要执行的program。如果未提供， 那么将使用 ``default_main_program``。 默认为None
    - **share_vars_from** (ParallelExecutor) – 如果提供了该参数， 则该 ``ParallelExecutor`` 与指定的 ``ParallelExecutor`` 共享变量。默          认为空
    - **exec_strategy** (ExecutionStrategy) – ``exec_strategy`` 用于调控program在 ``ParallelExecutor`` 中的执行方式，例如，执行该program需要的线程数, 释放在执行过程中产生的临时变量需要的重复(iterations)次数。 请参考 ``fluid.ExecutionStrategy`` 获取详细介绍。该参数默认为 None
    - **build_strategy** (BuildStrategy) – 设置成员 ``build_strategy`` 可以控制在 ``ParallelExecutor`` 中搭建SSA Graph的方式，例如， ``reduce_strategy`` ， ``gradient_scale_strategy`` 。 请参考 ``fluid.BuildStrategy`` 获取详细介绍。 该参数默认为None
    - **num_trainers** (int) – 如果该值大于1， NCCL将会通过多层级node的方式来初始化。每个node应有相同的GPU数目。 随之会启用分布式训练。该参数默认为1
    - **trainer_id** (int) – 必须与 ``num_trainers`` 参数同时使用。``trainer_id`` 是当前所在node的 “rank”（层级），从0开始计数。该参数默认为0
    - **scope** (Scope) – 指定执行program所在的作用域， 默认使用 ``fluid.global_scope()``

返回：初始化后的 ``ParallelExecutor`` 对象

返回类型: ParallelExecutor

抛出异常：``TypeError`` - 如果提供的参数 ``share_vars_from`` 不是 ``ParallelExecutor`` 类型的，将会弹出此异常

.. py:method::  run(fetch_list, feed=None, feed_dict=None, return_numpy=True)

使用 ``fetch_list`` 执行一个 ``ParallelExecutor`` 对象。

参数 ``feed`` 可以是 ``dict`` 或者 ``list`` 类型变量。如果该参数是 ``dict`` 类型，feed中的数据将会被分割(split)并分送给多个设备（CPU/GPU）。
反之，如果它是 ``list`` ，则列表中的各个元素都会直接分别被拷贝到各设备中。

**示例代码**

.. code-block:: python
    
    import paddle.fluid as fluid
    import numpy
    import os

    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
     
    # 注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
    # 否则fluid会把逻辑核的所有数目设为CPU_NUM，
    # 在这种情况下，输入的batch size应大于CPU_NUM，
    # 否则程序会异常中断。
    if not use_cuda:
        os.environ['CPU_NUM'] = str(2)
    exe = fluid.Executor(place)

    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
 
        startup_program.random_seed=1
        exe.run(startup_program)
 
        train_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                           main_program=train_program,
                                           loss_name=loss.name)
    # 如果feed参数是dict类型:
    # 图像会被split到设备中。假设有两个设备，那么每个设备将会处理形为 (5, 1)的图像
    x = numpy.random.random(size=(10, 1)).astype('float32')
    loss_data, = train_exe.run(feed={"X": x},

                               fetch_list=[loss.name])

    # 如果feed参数是list类型:
    # 各设备挨个处理列表中的每个元素
    # 第一个设备处理形为 (10, 1) 的图像
    # 第二个设备处理形为 (9, 1) 的图像
    #
    # 使用 exe.device_count 得到设备数目
    x2 = numpy.random.random(size=(9, 1)).astype('float32')
    loss_data, = train_exe.run(feed=[{"X": x}, {"X": x2}],
                               fetch_list=[loss.name])

参数：
    - **fetch_list** (list) – 获取的变量名列表
    - **feed** (list|dict|None) – feed变量。 如果该参数是 ``dict`` 类型，feed中的数据将会被分割(split)并分送给多个设备（CPU/GPU）。反之，如果它是 ``list`` ，则列表中的各个元素都直接分别被拷贝到各设备中。默认为None
    - **feed_dict** – 该参数已经停止使用。feed参数的别名, 为向后兼容而立。默认为None
    - **return_numpy** (bool) – 是否将fetched tensor转换为numpy。默认为True

返回： 获取的结果列表

返回类型：List

抛出异常:
     - ``ValueError`` - 如果feed参数是list类型，但是它的长度不等于可用设备（执行场所）的数目，再或者给定的feed不是dict类型，抛出此异常
     - ``TypeError`` - 如果feed参数是list类型，但是它里面的元素不是dict类型时，弹出此异常

.. note::
     1. 如果feed参数为dict类型，那么传入 ``ParallelExecutor`` 的数据量 *必须* 大于可用的CPU核数或GPU卡数。否则，C++端将会抛出异常。应额外注意核对数据集的最后一个batch是否比可用的CPU核数或GPU卡数大。
     2. 如果可用的CPU核数或GPU卡数大于一个，则为每个变量最后获取的结果都是list类型，且这个list中的每个元素都是各CPU核或GPU卡上的变量

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        pe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                    loss_name=avg_cost.name,
                                    main_program=fluid.default_main_program())
        loss = pe.run(feed=feeder.feed(cur_batch),
                      fetch_list=[avg_cost.name]))

.. py:method::  drop_local_exe_scopes()

立即删除本地执行作用域。
 
在程序执行期间，生成中间结果被放置在本地执行作用域内，在某些模型中，这些中间结果的创建和删除较为费时。为了解决这个问题，ParallelExecutor在ExecutionStrategy中提供了可选项，如num_iteration_per_drop_scope，此选项指示在删除本地执行作用域之前要运行的迭代次数。 但在某些情况下，每次迭代都会产生不同的中间结果，这将导致本地执行作用域所需的内存逐渐增加。 如果你想在这个时候运行另一个程序，可能没有足够的存储空间，此时你应该删除其他程序的本地执行作用域。
     

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
            import numpy
            import os
     
            use_cuda = True
            # 注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
            # 否则fluid会把逻辑核的所有数目设为CPU_NUM，
            # 在这种情况下，输入的batch size应大于CPU_NUM，
            # 否则程序会异常中断。
            if not use_cuda:
                os.environ['CPU_NUM'] = str(2)
     
            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                data = fluid.layers.data(name='X', shape=[1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
     
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe.run(startup_program)
     
            parallel_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                               main_program=train_program,
                                               loss_name=loss.name)
     
            x = numpy.random.random(size=(10, 1)).astype('float32')
            loss_data, = parallel_exe.run(feed={"X": x},
                                       fetch_list=[loss.name])
     
            parallel_exe.drop_local_exe_scopes()




