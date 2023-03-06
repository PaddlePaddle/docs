.. _cn_api_fluid_layers_py_func:

py_func
-------------------------------


.. py:function:: paddle.static.py_func(func, x, out, backward_func=None, skip_vars_in_backward_input=None)




PaddlePaddle 通过 py_func 在 Python 端注册 OP。py_func 的设计原理在于 Paddle 中的 Tensor 与 numpy 数组可以方便的互相转换，从而可使用 Python 中的 numpy API 来自定义一个 Python OP。

该自定义的 Python OP 的前向函数是 ``func``，反向函数是 ``backward_func`` 。 Paddle 将在前向部分调用 ``func``，并在反向部分调用 ``backward_func`` （如果 ``backward_func`` 不是 None)。 ``x`` 为 ``func`` 的输入，必须为 Tensor 类型；``out``  为 ``func`` 的输出，既可以是 Tensor 类型，也可以是 numpy 数组。

反向函数 ``backward_func`` 的输入依次为：前向输入 ``x`` 、前向输出 ``out`` 、 ``out`` 的梯度。如果 ``out`` 的某些输出没有梯度，则 ``backward_func`` 的相关输入为 None。如果 ``x`` 的某些变量没有梯度，则用户应在 ``backward_func`` 中主动返回 None。

在调用该接口之前，还应正确设置 ``out`` 的数据类型和形状，而 ``out`` 和 ``x`` 对应梯度的数据类型和形状将自动推断而出。

此功能还可用于调试正在运行的网络，可以通过添加没有输出的 ``py_func`` 运算，并在 ``func`` 中打印输入 ``x`` 。

参数
::::::::::::

    - **func** （callable） - 所注册的 Python OP 的前向函数，运行网络时，将根据该函数与前向输入 ``x``，计算前向输出 ``out``。在 ``func`` 建议先主动将 Tensor 转换为 numpy 数组，方便灵活的使用 numpy 相关的操作，如果未转换成 numpy，则可能某些操作无法兼容。
    - **x** (Tensor|tuple(Tensor)|list[Tensor]) -  前向函数 ``func`` 的输入，多个 Tensor 以 tuple(Tensor)或 list[Tensor]的形式传入。
    - **out** (T|tuple(T)|list[T]) -  前向函数 ``func`` 的输出，可以为 T|tuple(T)|list[T]，其中 T 既可以为 Tensor，也可以为 numpy 数组。由于 Paddle 无法自动推断 ``out`` 的形状和数据类型，必须应事先创建 ``out`` 。
    - **backward_func** (callable，可选) - 所注册的 Python OP 的反向函数。默认值为 None，意味着没有反向计算。若不为 None，则会在运行网络反向时调用 ``backward_func`` 计算 ``x`` 的梯度。
    - **skip_vars_in_backward_input** (Tensor，可选) -  ``backward_func`` 的输入中不需要的变量，可以是 Tensor|tuple(Tensor)|list[Tensor]。这些变量必须是 ``x`` 和 ``out`` 中的一个。默认值为 None，意味着没有变量需要从 ``x`` 和 ``out`` 中去除。若不为 None，则这些变量将不是 ``backward_func`` 的输入。该参数仅在 ``backward_func`` 不为 None 时有用。

返回
::::::::::::

Tensor|tuple(Tensor)|list[Tensor]，前向函数的输出 ``out``


代码示例 1
::::::::::::

COPY-FROM: paddle.static.py_func:code-example1


代码示例 2
::::::::::::

COPY-FROM: paddle.static.py_func:code-example2
