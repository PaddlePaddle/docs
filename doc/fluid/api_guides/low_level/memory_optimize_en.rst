.. _api_guide_memory_optimize_en:

#########################
Video Memory Optimization
#########################

**This guide is for GPU training.**

Video Memory optimization is to reduce the video memory consumption of :code:`Program` during execution by analyzing and reusing the video memory occupied by :code:`Variable` in :code:`Program`. Users can use the :code:`memory_optimize` interface to perform video memory optimization through Python scripts. The execution strategy of video memory optimization is as follows:

- Firstly, analyze the remaining lifetime of :code:`Variable` according to the relationship between :code:`Operator` in :code:`Program` to get the remaining lifetime of each :code:`Variable`;
- Secondly, according to the remaining lifetime of each :code:`Variable`, the future :code:`Variable` will reuse the video memory which is used by the :code:`Variable` that approches the end of its remaining lifetime or ceases to exist.

.. code-block:: python

	z = fluid.layers.sum([x, y])
	m = fluid.layers.matmul(y, z)

In this example, the lifetime of :code:`x` lasts until :code:`fluid.layers.sum`, so its video memory can be reused by :code:`m`.

Disable video memory optimization for specific parts
=======================================================

:code:`memory_optimize` supports disabling video memory optimization for specific sections. You can specify the :code:`Variable` whose video memory space is not going to be reused by passing in a collection of :code:`Variable` names;
At the same time, :code:`memory_optimize` disables video memory optimization for the backward part of the network, and the user can enable this function by passing in the :code:`skip_grads` parameter.

.. code-block:: python

	fluid.memory_optimize(fluid.default_main_program(),
		skip_opt_set=("fc"), skip_grads=True)

In this example, the :code:`fluid.memory_optimize` interface performs analysis of remaining lifetime of :code:`Variable` for the default :code:`Program`   , and skips the :code:`Variable` with the name :code:`fc` and all the :code:`Variable` in the backward part of the network .
This part of the :code:`Variable` video memory will not be used again by other :code:`Variable`.

Specify the video memory optimization level
==============================================

:code:`memory_optimize` supports printing information for video memory reusing to facilitate debugging. Users can enable debugging video memory reusing by specifying :code:`print_log=True`;

:code:`memory_optimize` supports two levels of video memory optimization, namely :code:`0` or :code:`1` :

- When the optimization level is :code:`0`: After :code:`memory_optimize` analyzes the remaining lifetime of :code:`Variable`, it will judge the :code:`shape` of :code:`Variable` . Memory reusing can only happens to the :code:`Variable` with the same :code:`shape`;
- When the optimization level is :code:`1`: the :code:`memory_optimize` will perform video memory reusing as much as possible. After analyzing the remaining survival time of :code:`Variable`, even with different :code:`shape`, the  :code:`Variable` will also perform the maximum amount of video memory reusing.

.. code-block:: python

	fluid.memory_optimize(fluid.default_main_program(),
		level=0, print_log=True)

In this example, the :code:`fluid.memory_optimize` interface performs analysis of remaining lifetime of :code:`Variable` for the default :code:`Program`   . Only when the :code:`shape` is exactly the same, will the :code:`Variable` enjoy video memory reusing. After the analysis is finished, all the debugging information related to video memory reusing will be printed out.
