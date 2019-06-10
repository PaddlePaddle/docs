..  _api_guide_executor:

##########
执行引擎
##########

:code:`Executor` 即 :code:`执行器` 。PaddlePaddle Fluid中有两种执行器可以选择。
:code:`Executor` 实现了一个简易的执行器，所有Operator会被顺序执行。用户可以使用
Python脚本驱动 :code:`Executor` 执行。默认情况下 :code:`Executor` 是单线程的，如果
想使用数据并行，请参考另一个执行器， :ref:`api_guide_parallel_executor` 。

:code:`Executor` 的代码逻辑非常简单。建议用户在调试过程中，先使用
:code:`Executor` 跑通模型，再切换到多设备计算，甚至多机计算。

:code:`Executor` 在构造的时候接受一个 :code:`Place`， 它们可以是 :ref:`cn_api_fluid_CPUPlace`
或 :ref:`cn_api_fluid_CUDAPlace` 。 :code:`Executor` 在执行的时候可以选择执行的
:ref:`api_guide_low_level_program` 。

简单的使用方法，请参考 `quick_start_fit_a_line <http://paddlepaddle.org/documentation/docs/zh/1.1/beginners_guide/quick_start/fit_a_line/README.cn.html>`_ , API Reference 请参考
:ref:`cn_api_fluid_Executor` 。
