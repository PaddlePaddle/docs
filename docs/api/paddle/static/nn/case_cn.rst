.. _cn_api_fluid_layers_case:

case
-------------------------------


.. py:function:: paddle.static.nn.case(pred_fn_pairs, default=None, name=None)


该OP的运行方式类似于python的if-elif-elif-else。

参数
::::::::::::

    - **pred_fn_pairs** (list|tuple) - 一个list或者tuple，元素是二元组(pred, fn)。其中 ``pred`` 是形状为[1]的布尔型 Tensor，``fn`` 是一个可调用对象。所有的可调用对象都返回相同结构的Tensor。
    - **default** (callable，可选) - 可调用对象，返回一个或多个张量。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor|list(Tensor)

- 如果 ``pred_fn_pairs`` 中存在pred是True的元组(pred, fn)，则返回第一个为True的pred的元组中fn的返回结果；如果 ``pred_fn_pairs`` 中不存在pred为True的元组(pred, fn) 且 ``default`` 不是None，则返回调用 ``default`` 的返回结果；
- 如果 ``pred_fn_pairs`` 中不存在pred为True的元组(pred, fn) 且 ``default`` 是None，则返回 ``pred_fn_pairs`` 中最后一个pred的返回结果。


代码示例
::::::::::::

COPY-FROM: paddle.static.nn.case