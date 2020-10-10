.. _cn_api_fluid_layers_case:

case
-------------------------------


.. py:function:: paddle.fluid.layers.case(pred_fn_pairs, default=None, name=None)

:api_attr: 声明式编程模式（静态图)


该OP的运行方式类似于python的if-elif-elif-else。

参数：
    - **pred_fn_pairs** (list|tuple) - 一个list或者tuple，元素是二元组(pred, fn)。其中 ``pred`` 是形状为[1]的布尔型 Tensor，``fn`` 是一个可调用对象。所有的可调用对象都返回相同结构的Tensor。
    - **default** (callable，可选) - 可调用对象，返回一个或多个张量。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值：None。

返回：如果 ``pred_fn_pairs`` 中存在pred是True的元组(pred, fn)，则返回第一个为True的pred的元组中fn的返回结果；如果 ``pred_fn_pairs`` 中不存在pred为True的元组(pred, fn) 且 ``default`` 不是None，则返回调用 ``default`` 的返回结果；
如果 ``pred_fn_pairs`` 中不存在pred为True的元组(pred, fn) 且 ``default`` 是None，则返回 ``pred_fn_pairs`` 中最后一个pred的返回结果。

返回类型：Tensor|list(Tensor)

抛出异常：
    - ``TypeError`` - 如果 ``pred_fn_pairs`` 的类型不是list或tuple。
    - ``TypeError`` - 如果 ``pred_fn_pairs`` 的元素的类型不是tuple。
    - ``TypeError`` - 如果 ``pred_fn_pairs`` 的tuple类型的元素大小不是2。
    - ``TypeError`` - 如果 ``pred_fn_pairs`` 中的2-tuple的第一个元素的类型不是Tensor。
    - ``TypeError`` - 如果 ``pred_fn_pairs`` 中的2-tuple的第二个元素不是可调用对象。
    - ``TypeError`` - 当 ``default`` 不是None又不是可调用对象时。

**代码示例**：

.. code-block:: python

    import paddle

    paddle.enable_static()

    def fn_1():
        return paddle.fill_constant(shape=[1, 2], dtype='float32', value=1)

    def fn_2():
        return paddle.fill_constant(shape=[2, 2], dtype='int32', value=2)

    def fn_3():
        return paddle.fill_constant(shape=[3], dtype='int32', value=3)

    main_program = paddle.static.default_startup_program()
    startup_program = paddle.static.default_main_program()

    with paddle.static.program_guard(main_program, startup_program):
        x = paddle.fill_constant(shape=[1], dtype='float32', value=0.3)
        y = paddle.fill_constant(shape=[1], dtype='float32', value=0.1)
        z = paddle.fill_constant(shape=[1], dtype='float32', value=0.2)

        pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3
        pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
        pred_3 = paddle.equal(x, y)      # false: 0.3 == 0.1

        # Call fn_1 because pred_1 is True
        out_1 = paddle.static.nn.case(
            pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3)

        # Argument default is None and no pred in pred_fn_pairs is True. fn_3 will be called.
        # because fn_3 is the last callable in pred_fn_pairs.
        out_2 = paddle.static.nn.case(pred_fn_pairs=[(pred_2, fn_2), (pred_3, fn_3)])

        exe = paddle.static.Executor(paddle.CPUPlace())
        res_1, res_2 = exe.run(main_program, fetch_list=[out_1, out_2])
        print(res_1)  # [[1. 1.]]
        print(res_2)  # [3 3 3]
