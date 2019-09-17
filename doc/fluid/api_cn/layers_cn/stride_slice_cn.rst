.. _cn_api_fluid_layers_strided_slice:

strided_slice
-------------------------------

.. py:function:: paddle.fluid.layers.strided_slice(input, axes, starts, ends, strides=strides)

strided_slice算子。

我们可以借助numpy数组的索引行为理解这个OP，如果你对numpy数组熟悉的话，应该知道通过[start1：end1：step1，start2：end2：step2，... startN：endN：stepN]的语法可以进行切片，从而通过一种十分简洁的方式获取数组的某些元素。strided_slice使得我们可以通过Paddle语法的方式完成相应的切割并获取相应的元素。例如numpy中input[start1:end1:step1, start2:end2:step2, ... startN:endN:stepN]可以通过Paddle相应的API函数paddle.fluid.strided_slice(input,[0, 1, ..., N], [starts1, starts2, ..., startsN],[ends1, ends2, ..., endsN], [strides1, sttides2, ..., stridesN])完成。axes参数用于控制对应的切割维度。
::

        案例1：
                给定：
                     data=[[1,2,3,4],[5,6,7,8],]
                     axes=[0,1]
                     starts=[1,0]
                     ends=[2,3]
                     strides=[1,1]
                则：
                     result=[[5,6,7],]

        案例2：
                给定：
                     data=[[1,2,3,4],[5,6,7,8],]
                     starts=[0,-1]
                     ends=[1,0]
                     strides=[1, -1]
                则：
                     result=[[4,3,2],]

参数：
        - **input** （Variable）- 提取切片的数据张量（Tensor）。
        - **axes** （List）- （list <int>）开始和结束的轴对应的维度。
        - **starts** （List）- （list <int>）在轴上开始相应轴的索引。
        - **ends** （List）- （list <int>）在轴上结束相应轴的索引。
        - **strides** （List）- （list <int>）在对应轴维度上的切割步长。

返回：        切片数据张量（Tensor）。

返回类型：        输出（Variable）。


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid

    starts = [1, 0, 2]
    ends = [3, 3, 4]
    axes = [0, 1, 2]
    strided_slice = [1, 1, 1]

    input = fluid.layers.data(
        name="input", shape=[3, 4, 5, 6], dtype='float32')

    out = fluid.layers.slice(input, axes=axes, starts=starts, ends=ends, strides=strides)
