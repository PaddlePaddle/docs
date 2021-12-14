paddle_fluid_IPUStrategy:

IPUStrategy
-------------------------------


.. py:class:: paddle.fluid.IPUStrategy

:api_attr: 声明式编程模式（静态图)



``IPUStrategy`` 使用户更方便地控制 :ref:`cn_api_fluid_IPUCompiledProgram` 中计算图的建造方法，可通过设置 ``compiler.py`` 中的 ``get_ipu_strategy`` 成员来获取对象。

代码示例
::::::::::

.. code-block:: python

    # required: ipu

    import paddle
    import paddle.fluid.compiler as compiler
    paddle.enable_static()
    main_prog = paddle.static.default_main_program()
    ipu_strategy = compiler.get_ipu_strategy()

.. py:attribute:: num_ipus

int类型。指定使用几个IPU，默认值为1。
    
代码示例
::::::::::

.. code-block:: python

    # required: ipu

    import paddle
    import paddle.fluid.compiler as compiler
    paddle.enable_static()
    main_prog = paddle.static.default_main_program()
    ipu_strategy = compiler.get_ipu_strategy()
    ipu_strategy.num_ipus = 1


.. py:attribute:: accumulationFactor

int类型。指定micro-batches的大小（等于导出的onnx模型的batch大小），默认值为1。
    
代码示例
::::::::::

.. code-block:: python

    # required: ipu

    import paddle
    import paddle.fluid.compiler as compiler
    paddle.enable_static()
    main_prog = paddle.static.default_main_program()
    ipu_strategy = compiler.get_ipu_strategy()
    ipu_strategy.accumulationFactor = 1

.. py:attribute:: batches_per_step

int类型。popart的概念，一次计算多少个batch，默认值为1。
    
代码示例
::::::::::

.. code-block:: python

    # required: ipu

    import paddle
    import paddle.fluid.compiler as compiler
    paddle.enable_static()
    main_prog = paddle.static.default_main_program()
    ipu_strategy = compiler.get_ipu_strategy()
    ipu_strategy.batches_per_step = 1

.. py:attribute:: is_training

bool类型。是否为训练模式，默认值为True。
    
代码示例
::::::::::

.. code-block:: python

    # required: ipu

    import paddle
    import paddle.fluid.compiler as compiler
    paddle.enable_static()
    main_prog = paddle.static.default_main_program()
    ipu_strategy = compiler.get_ipu_strategy()
    ipu_strategy.is_training = True

.. py:attribute:: enable_pipelining

bool类型。指定是否使用流水线模式，默认值为否。
    
代码示例
::::::::::

.. code-block:: python

    # required: ipu

    import paddle
    import paddle.fluid.compiler as compiler
    paddle.enable_static()
    main_prog = paddle.static.default_main_program()
    ipu_strategy = compiler.get_ipu_strategy()
    ipu_strategy.enable_pipelining = True

.. py:attribute:: enable_manual_shard

bool类型。指定是否为手动切分模型模式。
    
代码示例
::::::::::

.. code-block:: python

    # required: ipu

    import paddle
    import paddle.fluid.compiler as compiler
    paddle.enable_static()
    main_prog = paddle.static.default_main_program()
    ipu_strategy = compiler.get_ipu_strategy()
    ipu_strategy.enable_manual_shard = True

.. py:attribute:: need_avg_shard

bool类型。指定是否使用平均切分，调试时使用，默认值为否。
    
代码示例
::::::::::

.. code-block:: python

    # required: ipu

    import paddle
    import paddle.fluid.compiler as compiler
    paddle.enable_static()
    main_prog = paddle.static.default_main_program()
    ipu_strategy = compiler.get_ipu_strategy()
    ipu_strategy.need_avg_shard = True

代码示例
::::::::::

.. py:attribute:: batch_size

int类型。指定batch大小，默认为1。

.. code-block:: python

    # required: ipu

    import paddle
    import paddle.fluid.compiler as compiler
    paddle.enable_static()
    main_prog = paddle.static.default_main_program()
    ipu_strategy = compiler.get_ipu_strategy()
    ipu_strategy.batch_size = 1

.. py:attribute:: enable_fp16

bool类型。指定是否为fp16模式。

代码示例
::::::::::

.. code-block:: python

    # required: ipu

    import paddle
    import paddle.fluid.compiler as compiler
    paddle.enable_static()
    main_prog = paddle.static.default_main_program()
    ipu_strategy = compiler.get_ipu_strategy()
    ipu_strategy.enable_fp16 = True

