.. _cn_api_paddle_nn_Sequential:

Sequential
-------------------------------

.. py:class:: paddle.nn.Sequential(*layers)

顺序容器。子 Layer 将按构造函数参数的顺序依次添加到此容器中。
参数支持以下三种形式：

    1. 逐个 Layer 实例：Sequential(layer1, layer2, ...)
    2. 带名字的 (name, layer) 对：Sequential(('l1', layer1), ('l2', layer2))
    3. OrderedDict：Sequential(OrderedDict([('l1', layer1), ('l2', layer2)]))

可通过 整数下标 或 字符串名字 访问、修改、新增子层；支持 append、insert、extend 等列表式操作。


参数
::::::::::::

    - **layers** (Layer|tuple(str, Layer)|list|OrderedDict) - 任意数量的 Layer 对象，或 (name, layer) 可迭代结构，或一个 OrderedDict。

返回
::::::::::::
无

代码示例
::::::::::::

COPY-FROM: paddle.nn.Sequential
