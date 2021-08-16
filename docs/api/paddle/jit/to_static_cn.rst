.. _cn_api_paddle_jit_to_static:

to_static
-------------------------------

.. py:decorator:: paddle.jit.to_static

本装饰器将函数内的动态图API转化为静态图API。此装饰器自动处理静态图模式下的Program和Executor，并将结果作为动态图Tensor返回。输出的动态图Tensor可以继续进行动态图训练、预测或其他运算。如果被装饰的函数里面调用其他动态图函数，被调用的函数也会被转化为静态图函数。


参数：
    - **function** (callable) - 待转换的动态图函数。若以装饰器形式使用，则被装饰函数默认会被解析为此参数值，无需显式指定。
    - **input_spec** (list[InputSpec]|tuple[InputSpec]) - 用于指定被装饰函数中输入 Tensor 的 shape、dtype 和 name 信息，为包含 InputSpec 的 list/tuple 类型。
    - **build_strategy** (BuildStrategy|None): 通过配置 build_strategy，对转换后的计算图进行优化，例如：计算图中算子融合、计算图执行过程中开启内存/显存优化等。关于 build_strategy 更多信息，请参阅  ``paddle.static.BuildStrategy`` 。 默认为 None。


**示例代码**

.. code-block:: python

    import paddle
    from paddle.jit import to_static

    @to_static
    def func(x):
        if paddle.mean(x) < 0:
            x_v = x - 1
        else:
            x_v = x + 1
        return x_v

    x = paddle.ones([1, 2], dtype='float32')
    x_v = func(x)
    print(x_v) # [[2. 2.]]

