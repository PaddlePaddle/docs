..  _api_guide_executor_en:

################
Executor
################

:code:`Executor` realizes a simple executor in which all operators will be executed in order. You can run :code:`Executor` in a Python script. There are two kinds of executors in PaddlePaddle Fluid. One is single-thread executor which is the default option for :code:`Executor` 
and another is the parallel executor which is illustrated in :ref:`api_guide_parallel_executor_en` .

The logic of :code:`Executor` is very simple. It is suggested to thoroughly run the model with :code:`Executor` in debugging phase on one computer and then switch to mode of multiple devices or multiple computers to compute.

:code:`Executor` receives a :code:`Place` at construction, which can either be :ref:`api_fluid_CPUPlace` or :ref:`api_fluid_CUDAPlace`. 

For simple example please refer to `quick_start_fit_a_line <http://paddlepaddle.org/documentation/docs/zh/1.1/beginners_guide/quick_start/fit_a_line/README.html>`_ 

For API Reference, please refer to :ref:`api_fluid_Executor`.