.. _cn_api_fluid_layers_switch_case:

switch_case
-------------------------------


.. py:function:: paddle.static.nn.switch_case(branch_index, branch_fns, default=None, name=None)


运行方式类似于 c++的 switch/case。

参数
::::::::::::

    - **branch_index** (Tensor) - 元素个数为 1 的 Tensor （ 0-D Tensor 或者形状为 [1] ），指定将要执行的分支。数据类型是 ``int32``, ``int64`` 或 ``uint8``。
    - **branch_fns** (dict|list|tuple) - 如果 ``branch_fns`` 是一个 list 或 tuple，它的元素可以是 (int, callable) 二元组，即由整数和可调用对象构成的二元组，整数表示对应的可调用对象的键；也可以仅仅是可调用对象，它在 list 或者 tuple 中的实际索引值将作为该可调用对象的键。如果 ``branch_fns`` 是一个字典，那么它的键是整数，它的值是可调用对象。所有的可调用对象都返回相同结构的 Tensor。
    - **default** (callable，可选) - 可调用对象，返回一个或多个 Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor|list(Tensor)

- 如果 ``branch_fns`` 中存在与 ``branch_index`` 匹配的可调用对象，则返回该可调用对象的返回结果；如果 ``branch_fns`` 中不存在与 ``branch_index`` 匹配的可调用对象且 ``default`` 不是 None，则返回调用 ``default`` 的返回结果；
- 如果 ``branch_fns`` 中不存在与 ``branch_index`` 匹配的可调用对象且 ``default`` 是 None，则返回 ``branch_fns`` 中键值最大的可调用对象的返回结果。

代码示例
::::::::::::

COPY-FROM: paddle.static.nn.switch_case
