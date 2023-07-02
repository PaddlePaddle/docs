.. _cn_api_fluid_layers_cond:

cond
-------------------------------


.. py:function:: paddle.static.nn.cond(pred, true_fn=None, false_fn=None, name=None, return_names=None)


如果 ``pred`` 是 ``True``，该 API 返回 ``true_fn()``，否则返回 ``false_fn()``。
用户如果不想在 ``callable`` 中做任何事，可以把 ``true_fn`` 或 ``false_fn`` 设为 ``None``，此时本 API 会把该 ``callable`` 视为简单返回 ``None``。

``true_fn`` 和 ``false_fn`` 需要返回同样嵌套结构（nest structure）的 Tensor，如果不想返回任何值也可都返回 ``None``。
PaddlePaddle 里 Tensor 的嵌套结构是指一个 Tensor，或者 Tensor 的元组（tuple），或者 Tensor 的列表（list）。

.. note::
    1. ``true_fn`` 和 ``false_fn`` 返回的元组必须形状相同，但是里面的 Tensor 形状可以不同。
    2. 本接口在动态图和静态图模式下都可以运行，在动态图情况下就只会按 ``pred`` 条件运行其中一支分支。
    3. 静态图模式下，因为各个分支都要参与组网，因此不论运行哪个分支，在 ``true_fn`` 和 ``false_fn`` 内外创建的 Tensor 和 Op 都会组网，即 PaddlePaddle 并不是惰性语法（lazy semantics）。例如

       COPY-FROM: paddle.static.nn.cond:code-example-1

       不管 ``a < b`` 是否成立，``c = a * b`` 都会被组网且运行，``a + c`` 和 ``b * b`` 都会参与组网，只是组网后运行时只会运行条件对应的分支。

参数
:::::::::
    - **pred** (Tensor) - 一个元素个数为 1 的布尔型（boolean）的 Tensor （ 0-D Tensor 或者形状为 [1] ），该布尔值决定要返回 ``true_fn`` 还是 ``false_fn`` 的运行结果。
    - **true_fn** (callable) - 一个当 ``pred`` 是 ``True`` 时被调用的 callable，默认值：``None``。
    - **false_fn** (callable) - 一个当 ``pred`` 是 ``False`` 时被调用的 callable，默认值：``None``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值：``None``。
    - **return_names** (sequence of string，可选) - 通常情况下，用户不必设置此参数。字符串表示返回的变量的名称。序列的结构必须与 ``true_fn`` 和 ``false_fn`` 的返回值相同，默认值：``None``。

返回
:::::::::
Tensor|list(Tensor)|tuple(Tensor)，如果 ``pred`` 是 ``True``，该 API 返回 ``true_fn()``，否则返回 ``false_fn()``。

代码示例
:::::::::
COPY-FROM: paddle.static.nn.cond:code-example-2
