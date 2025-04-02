.. _api_paddle_logical_or__cn:

logical_or_
-------------------------------

.. py:function:: paddle.logical_or_(x: Tensor, y: Tensor, name: str|None = None) → Tensor

该 API 是 `paddle.logical_or` 的 **Inplace 版本**，输出结果将直接覆盖输入 Tensor ``x`` 的内存，功能与非 Inplace 版本一致。

逐元素对 ``x`` 和 ``y`` 进行逻辑或运算，计算公式为：

.. math::
    \text{out}_i = \text{x}_i \ \|\| \ \text{y}_i

.. note::
    **Inplace 警告**：此操作会直接修改输入 Tensor ``x`` 的值，若需保留原始数据，请使用非 Inplace 版本 `paddle.logical_or`。

    **广播机制**：``x`` 和 ``y`` 的形状需满足广播规则，详细规则请参考 `<../../guides/beginner/tensor_cn.html#id7>`_。

参数
:::::::::
- **x** (Tensor) - 输入的 Tensor，支持的数据类型为 bool、int8、int16、int32、int64、bfloat16、float16、float32、float64、complex64、complex128。
- **y** (Tensor) - 输入的 Tensor，数据类型需与 ``x`` 一致。
- **name** (str|None, 可选) - 操作的名称，默认值为 None，表示自动命名。

返回
:::::::::
    **Tensor**，与输入 ``x`` 共享内存的 Tensor，存储运算后的布尔值结果（直接覆盖 ``x``）。

代码示例
:::::::::

.. code-block:: python

    import paddle

    # 示例 1：基础用法
    x = paddle.to_tensor([True, False, True, False])
    y = paddle.to_tensor([True, True, False, False])
    paddle.logical_or_(x, y)
    print(x)  # 输出: [True, True, True, False]

    # 示例 2：广播机制
    x = paddle.to_tensor([[False], [True]])
    y = paddle.to_tensor([False, True, False, True]).reshape([2, 2])
    paddle.logical_or_(x, y)  # 广播为 [[False, True], [True, True]]
    print(x)
    # 输出: 
    # [[False, True],
    #  [True , True]]
