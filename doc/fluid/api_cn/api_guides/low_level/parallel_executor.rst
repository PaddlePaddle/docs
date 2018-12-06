.. _api_guide_parallel_executor:

#####
数据并行执行引擎
#####


:code:`ParallelExecutor` 是以数据并行的方式在多个节点上分别执行 :code:`Program` 的执行器。用户可以通过Python脚本驱动 :code:`ParallelExecutor` 执行， :code:`ParallelExecutor` 的执行过程：

- 首先根据 :code:`Program` 、 :code:`GPU` 卡的数目（或者 :code:`CPU` 的核数）以及 :ref:`api_fluid_BuildStrategy` 构建 :code:`SSA Graph` 和一个线程池;
- 执行过程中，根据Op的输入是否Ready决定是否执行该Op，这样可以使没有相互依赖的多个Op可在线程池中并行执行；

:code:`ParallelExecutor` 在构造时需要指定当前 :code:`Program` 的设备类型， :code:`GPU` 或者 :code:`CPU` ：

* 使用 :code:`GPU` 执行： :code:`ParallelExecutor` 会自动检测当前机器可以使用 :code:`GPU` 的个数，并在每个 :code:`GPU` 上分别执行 :code:`Program` ，用户也可以通过设置 :code:`CUDA_VISIBLE_DEVICES` 环境变量来指定执行器可使用的 :code:`GPU` ；
* 使用 :code:`CPU` 多线程执行：:code:`ParallelExecutor` 会自动检测当前机器可利用的 :code:`CPU` 核数，并将 :code:`CPU` 核数作为执行器中线程的个数，每个线程分别执行 :code:`Program` ，用户也可以通过设置 :code:`CPU_NUM` 环境变量来指定当前训练使用的线程个数。

:code:`ParallelExecutor` 支持模型训练和模型预测：

* 模型训练： :code:`ParallelExecutor` 在执行过程中对多个节点上的参数梯度进行聚合，然后进行参数的更新；
* 模型预测： :code:`ParallelExecutor` 在执行过程中各个节点独立运行当前的  :code:`Program` ；

:code:`ParallelExecutor` 在模型训练时支持两种模式的梯度聚合, :code:`AllReduce` 和 :code:`Reduce` ：

* :code:`AllReduce` 模式下， :code:`ParallelExecutor` 调用AllReduce操作使多个节点上参数梯度完全相等，然后各个节点独立进行参数的更新；
* :code:`Reduce` 模式下， :code:`ParallelExecutor` 会预先将所有参数的更新分派到不同的节点上，在执行过程中 :code:`ParallelExecutor` 调用Reduce操作将参数梯度在预先指定的节点上进行聚合，并进行参数更新，最后调用Broadcast操作将更新后的参数发送到其他节点。

这两种模式通过 :code:`build_strategy` 来指定，使用方法，请参考 :ref:`api_fluid_BuildStrategy` 。

**注意** ：如果在Reduce模式下使用 :code:`CPU` 多线程执行 :code:`Program` ， :code:`Program` 的参数在多个线程间是共享的，在某些模型上，Reduce模式可以大幅节省内存。

由于模型的执行速度与模型结构和执行器的执行策略相关， :code:`ParallelExecutor` 允许用户修改执行器的相关参数，如：线程池大小（ :code:`num_threads` ）、多少次迭代之后清理一次临时变量 :code:`num_iteration_per_drop_scope` 等，更多信息请参考 :ref:`api_fluid_ExecutionStrategy` >。

- 相关API汇总:
 - :ref:`api_fluid_ParallelExecutor`
 - :ref:`api_fluid_BuildStrategy`
 - :ref:`api_fluid_ExecutionStrategy`