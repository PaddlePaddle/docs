.. _cn_api_fluid_CompiledProgram:

CompiledProgram
-------------------------------

.. py:class:: paddle.fluid.CompiledProgram(program_or_graph)

编译成一个用来执行的Graph。

1. 首先使用layers(网络层)创建程序。
2. （可选）可使用CompiledProgram来在运行之前优化程序。
3. 定义的程序或CompiledProgram由Executor运行。

CompiledProgram用于转换程序以进行各种优化。例如，

- 预先计算一些逻辑，以便每次运行更快。
- 转换Program，使其可以在多个设备中运行。
- 转换Program以进行优化预测或分布式训练。注意：此部分尚未完成。

**代码示例**

.. code-block:: python
        
        import paddle.fluid as fluid
        import paddle.fluid.compiler as compiler
        import numpy
        import os
     
        place = fluid.CUDAPlace(0) # fluid.CPUPlace()
        exe = fluid.Executor(place)
     
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
        loss = fluid.layers.mean(hidden)
        fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
     
        fluid.default_startup_program().random_seed=1
        exe.run(fluid.default_startup_program())
        compiled_prog = compiler.CompiledProgram(
                 fluid.default_main_program())
     
        x = numpy.random.random(size=(10, 1)).astype('float32')
        loss_data, = exe.run(compiled_prog,
                             feed={"X": x},
                             fetch_list=[loss.name])
参数：
  - **program_or_graph** (Graph|Program): 如果它是Program，那么它将首先被降成一个graph，以便进一步优化。如果它是一个graph（以前可能优化过），它将直接用于进一步的优化。注意：只有使用 with_data_parallel 选项编译时才支持graph。

.. py:method:: with_data_parallel(loss_name=None, build_strategy=None, exec_strategy=None, share_vars_from=None, places=None)

配置Program使其以数据并行方式运行。

**代码示例**

.. code-block:: python
            
            import paddle.fluid as fluid
            import paddle.fluid.compiler as compiler
            import numpy
            import os
     
            use_cuda = True
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            #注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
            #否则fluid会把逻辑核的所有数目设为CPU_NUM，
            #在这种情况下，输入的batch size应大于CPU_NUM，
            #否则程序会异常中断。
            if not use_cuda:
                os.environ['CPU_NUM'] = str(2)
     
            exe = fluid.Executor(place)
     
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
     
            fluid.default_startup_program().random_seed=1
            exe.run(fluid.default_startup_program())
            compiled_prog = compiler.CompiledProgram(
                     fluid.default_main_program()).with_data_parallel(
                              loss_name=loss.name)
     
            x = numpy.random.random(size=(10, 1)).astype('float32')
            loss_data, = exe.run(compiled_prog,
                                 feed={"X": x},
                                 fetch_list=[loss.name])
     
参数：
  - **loss_name** （str） - 损失函数名称必须在训练过程中设置。 默认None。
  - **build_strategy** （BuildStrategy） -  build_strategy用于构建图，因此它可以在具有优化拓扑的多个设备/核上运行。 有关更多信息，请参阅  ``fluid.BuildStrategy`` 。 默认None。
  - **exec_strategy** （ExecutionStrategy） -  exec_strategy用于选择执行图的方式，例如使用多少线程，每次清理临时变量之前进行的迭代次数。 有关更多信息，请参阅 ``fluid.ExecutionStrategy`` 。 默认None。
  - **share_vars_from** （CompiledProgram） - 如果有，此CompiledProgram将共享来自share_vars_from的变量。 share_vars_from指定的Program必须由此CompiledProgram之前的Executor运行，以便vars准备就绪。
  - **places** （list(CUDAPlace)|list(CPUPlace)|None） - 如果提供，则仅在给定位置编译程序。否则，编译时使用的位置由Executor确定，使用的位置由环境变量控制：如果使用GPU，则标记FLAGS_selected_gpus或CUDA_VISIBLE_DEVICES设备；如果使用CPU，则标记CPU_NUM。例如，如果要在GPU 0和GPU 1上运行，请设置places=[fluid.CUDAPlace(0), fluid.CUDAPlace(1)]。如果要在2个CPU核心上运行，请设置places=[fluid.CPUPlace()]*2。

返回: self

.. py:method:: with_inference_optimize(config)

添加预测优化。

参数：
  - **config** - 用于创建预测器的NativeConfig或AnalysisConfig的实例

返回: self


