.. _cn_api_paddle_static_data:

data
-------------------------------


.. py:function:: paddle.static.data(name, shape, dtype=None, lod_level=0)




会在全局 block 中创建变量（Tensor），该全局变量可被计算图中的算子（operator）访问。该变量可作为占位符用于数据输入。例如用执行器（Executor）输入数据进该变量，当 ``dtype`` 为 None 时，``dtype`` 将通过 ``padle.get_default_dtype()`` 获取全局类型。


参数
::::::::::::

    - **name** (str) - 变量名，具体用法请参见 :ref:`api_guide_Name`。
    - **shape** (list|tuple)- 声明维度信息的 list 或 tuple。可以在某个维度上设置 None 或-1，以指示该维度可以是任何大小。例如，将可变 batchsize 设置为 None 或-1。
    - **dtype** (np.dtype|str，可选)- 数据类型，支持 bool，float16，float32，float64，int8，int16，int32，int64，uint8。默认值为 None。当 ``dtype`` 为 None 时，``dtype`` 将通过 ``padle.get_default_dtype()`` 获取全局类型。
    - **lod_level** (int，可选)- LoDTensor 变量的 LoD level 数，LoD level 是 PaddlePaddle 的高级特性，一般任务中不会需要更改此默认值。默认值为 0。

返回
::::::::::::
Tensor，全局变量，可进行数据访问。


代码示例
::::::::::::

COPY-FROM: paddle.static.data
