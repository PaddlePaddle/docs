.. _api_guide_memory_optimize_en:

####################
Memory Optimization
####################

Memory optimization is a method to reduce the memory consumption of :code:`Program` during execution by analyzing and reusing the memory used by :code:`Varaible` in :code:`Program`. Users can use the :code:`memory_optimize` interface to perform memory optimization through Python scripts. The execution strategy of memory optimization is as follows:

- Firstly, analyze the last surviving time of :code:`Variable` according to the relationship between :code:`Operator` in :code:`Program`, and get the last surviving time of each :code:`Variable`;
- Secondly, according to the last surviving time of each :code:`Variable`, we will provide the memory that is used by the surviving time and no longer surviving's :code:`Variable` is provided to the later :code:`Variable`.

.. code-block:: python

	z = fluid.layers.sum([x, y])
	m = fluid.layers.matmul(y, z)

In this example, the surviving time of :code:`x` is up to :code:`fluid.layers.sum`'s operation, so its memory can be reused by :code:`m`.

Disable memory optimization for specific parts
================================================

:code:`memory_optimize` supports disabling memory optimization for specific sections. Users can specify which of the :code:`Variable` names are not reused by passing in a collection of :code:`Variable` names;
At the same time, :code:`memory_optimize` disables memory optimization for the reverse part of the network, and the user can enable this function by passing in the :code:`skip_grads` parameter.

.. code-block:: python

	fluid.memory_optimize(fluid.default_main_program(),
		skip_opt_set=("fc"), skip_grads=True)

In this example, the :code:`fluid.memory_optimize` interface for the default :code:`Program` is done  :code:`Variable` analysis of the last time to live, and skips the :code:`Variable` with the name :code:`fc` and all the reverse parts of the network :code:`Variable`.
This part of the :code:`Variable` memory will not be used again by other :code:`Varaible`.

Specify the memory optimization level
=======================================

:code:`memory_optimize` supports printing information for memory reuse to facilitate user debugging. Users can enable dubugging memory multiplexing by specifying :code:`print_log=True`;

:code:`memory_optimize` supports two levels of memory optimization, :code:`0` or :code:`1` :

- When the optimization level is :code:`0`: After :code:`memory_optimize` analyzing the last survival time of :code:`Variable`, it will judge :code:`Variable` 's :code:`shape`, :code:`Variable` will be only used for video memory multiplexing in the same :code:`shape`;
- When the optimization level is :code:`1`: ,the :code:`memory_optimize` will perform memory multiplexing as much as possible. After analyzing the last survival time of :code:`Variable`, even if different :code:`shape`, the  :code:`Variable` will also perform the maximum amount of memory multiplexing.

.. code-block:: python

	fluid.memory_optimize(fluid.default_main_program(),
		level=0, print_log=True)

In this example, the :code:`fluid.memory_optimize` interface performs an :code:`Variable` last surviving time analysis to the default :code:`Program` 
. Only the :code:`shape` is exactly the same, the :code:`Variable` will be used for memory multiplexing, and after the analysis is finished, all the debugging information related to memory sharing will be printed out.
