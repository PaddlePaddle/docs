.. _cn_api_paddle_nn_quant_Stub:
Stub
-------------------------------

.. py:class:: paddle.nn.quant.Stub(observer=None)

存根用作占位符，在 PTQ 或 QAT 之前将被观察者替换。将量化配置分配给在层的转发中调用的功能 API 是很困难的。相反，我们可以创建一个存根，并将其添加到层的子层。并在转发函数 API 之前调用存根。由存根持有的观察者将观察或量化功能 API 的输入。

参数
::::::::::::

    - **observer** (QuanterFactory) - 要插入的观察点的配置信息。如果 observer 为 none, 它将使用全局配置来创建 observer - 表示不使用观察器，也不收集任何关于模型参数或激活的统计数据，仅作为占位符存在。

代码示例
::::::::::::

COPY-FROM: paddle.nn.quant.stub

forward(input)
''''''''''''
定义每次调用时执行的计算。应由所有子类覆盖。

参数
::::::::::::
    - **input** (tuple) - 解压缩的元组参数。
    - **kwargs** (dict) - 解压缩的 dict 参数。
