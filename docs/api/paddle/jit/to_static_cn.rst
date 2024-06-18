.. _cn_api_paddle_jit_to_static:

to_static
-------------------------------

.. py:function:: paddle.jit.to_static

本装饰器将函数内的动态图 API 转化为静态图 API。此装饰器自动处理静态图模式下的 Program 和 Executor，并将结果作为动态图 Tensor 返回。输出的动态图 Tensor 可以继续进行动态图训练、预测或其他运算。如果被装饰的函数里面调用其他动态图函数，被调用的函数也会被转化为静态图函数。


参数
::::::::::::

    - **function** (callable) - 待转换的动态图函数。若以装饰器形式使用，则被装饰函数默认会被解析为此参数值，无需显式指定。
    - **input_spec** (list[InputSpec]|tuple[InputSpec]) - 用于指定被装饰函数中输入 Tensor 的 shape、dtype 和 name 信息，为包含 InputSpec 的 list/tuple 类型。
    - **build_strategy** (BuildStrategy|None)：通过配置 build_strategy，对转换后的计算图进行优化，例如：计算图中算子融合、计算图执行过程中开启内存/显存优化等。关于 build_strategy 更多信息，请参阅  ``paddle.static.BuildStrategy``。默认为 None。
    - **backend** (str，可选): 指定后端编译器，可以指定为 `CINN` 或者 None。当该参数指定为 `CINN` 时，将会使用 CINN 编译器来加速训练和推理。
    - **kwargs**: 支持的 key 包括 `property`

        - **property**: 表示被装饰的函数是否以 class property 属性的方式进行导出


代码示例
::::::::::::

COPY-FROM: paddle.jit.to_static
