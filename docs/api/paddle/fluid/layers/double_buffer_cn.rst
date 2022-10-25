.. _cn_api_fluid_layers_double_buffer:

double_buffer
-------------------------------

.. py:function:: paddle.fluid.layers.double_buffer(reader, place=None, name=None)





生成一个双缓冲队列Reader。Reader类有DecoratedReader和FileReader，其中DecoratedReader又可以细分成CustomReader和BufferedReader。这里是基于BufferedReader，数据将复制到具有双缓冲队列的位置（由place指定），如果 ``place=None`` 则将使用executor执行的位置。

参数
::::::::::::

    - **reader** (Variable) – 需要wrap的reader变量Reader。
    - **place** (Place，可选) – 目标数据的位置，比如CPU，GPU，GPU需要指明是哪张卡。默认是executor执行样本的位置。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。



返回
::::::::::::
Variable(Reader)。双缓冲队列的reader。

返回类型
::::::::::::
变量(Variable)。


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.double_buffer