.. _api_guide_memory_optimize:

#####
显存优化
#####

显存优化是通过分析、复用 :code:`Program` 中 :code:`Varaible` 使用的显存，从而降低 :code:`Program` 执行时显存消耗的方法。用户可以通过Python脚本调用 :code:`memory_optimize` 接口进行显存优化，显存优化的执行策略如下：

- 首先根据 :code:`Program` 中 :code:`Operator` 之间的关系对 :code:`Variable` 的最后存活时间进行分析，得到每个 :code:`Variable` 的最后存活时间;
- 其次根据每个 :code:`Variable` 的最后存活时间，我们将到达存活时间、不再存活的 :code:`Variable` 所占用的显存提供给后来的 :code:`Variable` 使用。

.. code-block:: python

    z = fluid.layers.sum([x, y])
    m = fluid.layers.matmul(y, z)

在这个示例中，:code:`x` 的存活时间到 :code:`fluid.layers.sum` 操作为止，所以它的显存可以被 :code:`m` 复用。

针对特定部分禁用显存优化
===========

:code:`memory_optimize` 支持针对特定部分禁用显存优化，用户可以通过传入 :code:`Variable` 名字的集合来指定哪些 :code:`Variable` 所使用的显存不会被复用;
与此同时，:code:`memory_optimize` 能够针对网络的反向部分禁用显存优化，用户可以通过传入 :code:`skip_grads` 参数来开启这个功能。

.. code-block:: python

    fluid.memory_optimize(fluid.default_main_program(),
        skip_opt_set=("fc"), skip_grads=True)

在这个示例中，:code:`fluid.memory_optimize` 接口对默认的 :code:`Program` 进行了 :code:`Variable` 最后存活时间的分析，并跳过了名字为 :code:`fc` 的 :code:`Variable` 以及网络反向部分的所有 :code:`Variable` 。
这部分 :code:`Variable` 的显存都不会被别的 :code:`Varaible` 再次使用。

指定显存优化等级
===========

:code:`memory_optimize` 支持打印显存复用的信息以方便用户进行调试，用户可以通过指定 :code:`print_log=True` 来开启显存复用的调试信息;

:code:`memory_optimize` 支持两种显存优化的等级，:code:`0` 或者 :code:`1` :

- 优化等级为 :code:`0` 时： :code:`memory_optimize` 在分析完 :code:`Variable` 的最后生存时间后，会判断 :code:`Variable` 的 :code:`shape` ，只有 :code:`shape` 相同的 :code:`Variable` 才会进行显存复用；
- 优化等级为 :code:`1` 时： :code:`memory_optimize` 会尽可能地进行显存复用，在分析完 :code:`Variable` 的最后生存时间后，即使是 :code:`shape` 不同的 :code:`Variable` 也会进行最大程度的显存复用。

.. code-block:: python

    fluid.memory_optimize(fluid.default_main_program(),
        level=0, print_log=True)

在这个示例中，:code:`fluid.memory_optimize` 接口对默认的 :code:`Program` 进行了 :code:`Variable` 最后存活时间的分析。
只有 :code:`shape` 完全相同的 :code:`Variable` 才会进行显存复用，并且在分析结束后，会打印出所有显存复用相关的调试信息。
