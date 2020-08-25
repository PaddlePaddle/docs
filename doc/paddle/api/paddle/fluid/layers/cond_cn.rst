.. _cn_api_fluid_layers_cond:

cond
-------------------------------


.. py:function:: paddle.fluid.layers.cond(pred, true_fn=None, false_fn=None, name=None)

:api_attr: 声明式编程模式（静态图)
:alias_main: paddle.nn.cond
:alias: paddle.nn.cond,paddle.nn.control_flow.cond
:old_api: paddle.fluid.layers.cond



如果 ``pred`` 是 ``True`` ，该API返回 ``true_fn()`` ，否则返回 ``false_fn()`` 。
用户如果不想在 ``callable`` 中做任何事，可以把 ``true_fn`` 或 ``false_fn`` 设为 ``None`` ，此时本API会把该 ``callable`` 视为简单返回 ``None`` 。

``true_fn`` 和 ``false_fn`` 需要返回同样嵌套结构（nest structure）的Tensor，如果不想返回任何值也可都返回 ``None`` 。
PaddlePaddle里Tensor的嵌套结构是指一个Tensor，或者Tensor的元组（tuple），或者Tensor的列表（list）。

.. note::
    1. 因为PaddlePaddle的静态图数据流， ``true_fn`` 和 ``false_fn`` 返回的元组必须形状相同，但是里面的Tensor形状可以不同。
    2. 不论运行哪个分支，在 ``true_fn`` 和 ``false_fn`` 外创建的Tensor和Op都会被运行，即PaddlePaddle并不是惰性语法（lazy semantics）。例如

       .. code-block:: python
                  
            import paddle.fluid as fluid
            a = fluid.data(name='a', shape=[-1, 1], dtype='float32')
            b = fluid.data(name='b', shape=[-1, 1], dtype='float32')
            c = a * b
            out = fluid.layers.cond(a < b, lambda: a + c, lambda: b * b)

       不管 ``a < b`` 是否成立， ``c = a * b`` 都会被运行。

参数：
    - **pred** (Variable) - 一个形状为[1]的布尔型（boolean）的Tensor，该布尔值决定要返回 ``true_fn`` 还是 ``false_fn`` 的运行结果。
    - **true_fn** (callable) - 一个当 ``pred`` 是 ``True`` 时被调用的callable，默认值： ``None`` 。
    - **false_fn** (callable) - 一个当 ``pred`` 是 ``False`` 时被调用的callable，默认值： ``None`` 。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值： ``None`` 。

返回：
    如果 ``pred`` 是 ``True`` ，该API返回 ``true_fn()`` ，否则返回 ``false_fn()`` 。

返回类型：Variable|list(Variable)|tuple(Variable)

抛出异常：
    - ``TypeError`` - 如果 ``true_fn`` 或 ``false_fn`` 不是callable。
    - ``ValueError`` - 如果 ``true_fn`` 和 ``false_fn`` 没有返回同样的嵌套结构（nest structure），对嵌套结构的解释见上文。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    from paddle.fluid.executor import Executor
    from paddle.fluid.framework import Program, program_guard

    #
    # pseudocode:
    # if 0.1 < 0.23:
    #     return 1, True
    # else:
    #     return 3, 2
    #

    def true_func():
        return layers.fill_constant(
            shape=[1, 2], dtype='int32', value=1), layers.fill_constant(
                shape=[2, 3], dtype='bool', value=True)

    def false_func():
        return layers.fill_constant(
            shape=[3, 4], dtype='float32', value=3), layers.fill_constant(
                shape=[4, 5], dtype='int64', value=2)

    main_program = Program()
    startup_program = Program()
    with program_guard(main_program, startup_program):
        x = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
        y = layers.fill_constant(shape=[1], dtype='float32', value=0.23)
        pred = layers.less_than(x, y)            
        out = layers.cond(pred, true_func, false_func)
        # out is a tuple containing 2 tensors

        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        ret = exe.run(main_program, fetch_list=out)
        # ret[0] = [[1 1]]
        # ret[1] = [[ True  True  True]
        #           [ True  True  True]]

