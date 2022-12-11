..  _api_guide_executor_en:

################
Executor
################

:code:`Executor` realizes a simple executor in which all operators will be executed in order. You can run :code:`Executor` in a Python script. There are two kinds of executors in PaddlePaddle Fluid. One is single-thread executor which is the default option for :code:`Executor` and the other is the parallel executor which is illustrated in :ref:`api_guide_parallel_executor_en` . The config of `Executor` and :ref:`api_guide_parallel_executor_en` is different, it may be a bit confusing for some users. To make the executor more facility, we introduce :ref:`api_guide_compiled_program_en` , :ref:`api_guide_compiled_program_en` is used to transform a program for various optimizations, and it can be run by :code:`Executor`.

The logic of :code:`Executor` is very simple. It is suggested to thoroughly run the model with :code:`Executor` in debugging phase on one computer and then switch to mode of multiple devices or multiple computers to compute.

:code:`Executor` receives a :code:`Place` at construction, which can either be :ref:`api_fluid_CPUPlace` or :ref:`api_fluid_CUDAPlace`.

.. code-block:: python

    # First create the Executor.
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Run the startup program once and only once.
    exe.run(fluid.default_startup_program())

    # Run the main program directly.
    loss, = exe.run(fluid.default_main_program(),
                    feed=feed_dict,
                    fetch_list=[loss.name])


For simple example please refer to `basics_fit_a_line <../../beginners_guide/basics/fit_a_line/README.html>`_

- Related API :
 - :ref:`api_fluid_Executor`
