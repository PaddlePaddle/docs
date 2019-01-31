..  _api_guide_executor_en:

################
Executor Engine
################

:code:`Executor` , :code:`执行器` in Chinese, is a simple executor with which all Operators will be executed in order. You can run :code:`Executor` driven by Python script. There are two executors in PaddlePaddle Fluid. One is single-thread executor which is the default option for :code:`Executor` .
and another is parallel executor which can be referred to :ref:`api_guide_parallel_executor` .

The logic of :code:`Executor` is very simple. It is suggested to thoroughly run the model with :code:`Executor` in debug in one computer and then change to multiple devices or multiple computers to compute.

:code:`Executor` receieve a :code:`Place` at construction, which can be either :ref:`api_fluid_CPUPlace` or :ref:`api_fluid_CUDAPlace`. :code:`Executor` can choose to run :ref:`api_guide_low_level_program`.

About quick start, please refer to `quick_start_fit_a_line <http://paddlepaddle.org/documentation/docs/zh/1.1/beginners_guide/quick_start/fit_a_line/README.cn.html>`_ 

About API Reference, please refer to :ref:`api_fluid_Executor`.