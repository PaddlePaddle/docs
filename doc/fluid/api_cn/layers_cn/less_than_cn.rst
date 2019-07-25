.. _cn_api_fluid_layers_less_than:

less_than
-------------------------------

.. py:function:: paddle.fluid.layers.less_than(x, y, force_cpu=None, cond=None)


该函数按元素出现顺序依次在X,Y上操作，并返回 ``Out`` ，它们三个都是n维tensor（张量）。
其中，X、Y可以是任何类型的tensor，Out张量的各个元素可以通过 :math:`Out=X<Y` 计算得出。


参数：
    - **x** (Variable) – ``less_than`` 运算的左操作数
    - **y** (Variable) – ``less_than`` 运算的右操作数
    - **force_cpu** (BOOLEAN) – 值True则强制将输出变量写入CPU内存中。否则，将其写入目前所在的运算设备上。默认为True
    - **cond** (Variable|None) – 可选的用于存储 ``less_than`` 输出结果的变量，为None则由函数自动生成Out变量


返回： n维bool型tensor，其中各个元素可以通过 *Out=X<Y* 计算得出

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name='y', shape=[1], dtype='int64')
    limit = fluid.layers.fill_constant(shape=[1], dtype='int64', value=5)
    cond = fluid.layers.less_than(x=label, y=limit)


