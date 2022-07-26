.. _cn_api_static_cn_data:

data
-------------------------------


.. py:function:: paddle.static.data(name, shape, dtype=None, lod_level=0)




该OP会在全局block中创建变量（Tensor），该全局变量可被计算图中的算子（operator）访问。该变量可作为占位符用于数据输入。例如用执行器（Executor）feed数据进该变量，当 ``dtype`` 为None时，``dtype`` 将通过 ``padle.get_default_dtype()`` 获取全局类型。


参数
::::::::::::

    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **shape** (list|tuple)- 声明维度信息的list或tuple。可以在某个维度上设置None或-1，以指示该维度可以是任何大小。例如，将可变batchsize设置为None或-1。
    - **dtype** (np.dtype|str，可选)- 数据类型，支持bool，float16，float32，float64，int8，int16，int32，int64，uint8。默认值为None。当 ``dtype`` 为None时，``dtype`` 将通过 ``padle.get_default_dtype()`` 获取全局类型。
    - **lod_level** (int，可选)- LoDTensor变量的LoD level数，LoD level是PaddlePaddle的高级特性，一般任务中不会需要更改此默认值，关于LoD level的详细适用场景和用法请见 :ref:`cn_user_guide_lod_tensor`。默认值为0。

返回
::::::::::::
Tensor，全局变量，可进行数据访问。


代码示例
::::::::::::

COPY-FROM: paddle.static.data