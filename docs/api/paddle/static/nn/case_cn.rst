.. _cn_api_fluid_layers_case:

case
-------------------------------


.. py:function:: paddle.static.nn.case(pred_fn_pairs, default=None, name=None)


运行方式类似于 python 的 if-elif-elif-else。

参数
::::::::::::

    - **pred_fn_pairs** (list|tuple) - 一个 list 或者 tuple，元素是二元组(pred, fn)。其中 ``pred`` 是元素个数为 1 的布尔型 Tensor （ 0D Tensor 或者形状为 [1] ），``fn`` 是一个可调用对象。所有的可调用对象都返回相同结构的 Tensor。
    - **default** (callable，可选) - 可调用对象，返回一个或多个 Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor|list(Tensor)

- 如果 ``pred_fn_pairs`` 中存在 pred 是 True 的元组(pred, fn)，则返回第一个为 True 的 pred 的元组中 fn 的返回结果；如果 ``pred_fn_pairs`` 中不存在 pred 为 True 的元组(pred, fn) 且 ``default`` 不是 None，则返回调用 ``default`` 的返回结果；
- 如果 ``pred_fn_pairs`` 中不存在 pred 为 True 的元组(pred, fn) 且 ``default`` 是 None，则返回 ``pred_fn_pairs`` 中最后一个 pred 的返回结果。


代码示例
::::::::::::

COPY-FROM: paddle.static.nn.case
