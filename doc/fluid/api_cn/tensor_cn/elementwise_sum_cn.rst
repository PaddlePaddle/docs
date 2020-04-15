.. _cn_api_tensor_elementwise_sum:

elementwise_sum
-------------------------------

.. py:function:: paddle.elementwise_sum(inputs, name=None)

该OP用于对输入的一至多个Tensor或LoDTensor求和。如果输入的是LoDTensor，输出仅与第一个输入共享LoD信息（序列信息）。

例1：
::
    输入：
        input.shape = [2, 3]
        input = [[1, 2, 3],
              [4, 5, 6]]

    输出：
        output.shape = [2, 3]
        output = [[1, 2, 3],
                [4, 5, 6]]

例2：
::
    输入：
    第一个输入：
            input1.shape = [2, 3]
            input1 = [[1, 2, 3],
                  [4, 5, 6]]

    第二个输入：
            input2.shape = [2, 3]
            input2 = [[7, 8, 9],
                  [10, 11, 12]]

    输出：
        output.shape = [2, 3]
        output = [[8, 10, 12],
              [14, 16, 18]]

参数：
    - **inputs** (Variable|list(Variable)) - 输入的一至多个Variable。如果输入了多个Variable，则不同Variable间的shape和数据类型应保持一致。Variable为多维Tensor或LoDTensor，数据类型支持：float32，float64，int32，int64。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：对输入 ``inputs`` 中的Variable求和后的结果，shape和数据类型与 ``inputs`` 一致。

返回类型：Variable


**代码示例：**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    input0 = fluid.layers.fill_constant(shape=[2, 3], dtype='int64', value=5)
    input1 = fluid.layers.fill_constant(shape=[2, 3], dtype='int64', value=3)
    sum = paddle.elementwise_sum([input0, input1])

    #用户可以通过executor打印出求和的结果
    out = fluid.layers.Print(sum, message="the sum of input0 and input1: ")
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_main_program())

    #打印出的数据为：
    1570701754  the sum of input0 and input1:   The place is:CPUPlace
    Tensor[elementwise_sum_0.tmp_0]
        shape: [2,3,]
        dtype: l
        data: 8,8,8,8,8,8,

    #输出了shape为[2,3]的Tensor，与输入的shape一致
    #dtype为对应C++数据类型，在不同环境下可能显示值不同，但本质相同
    #例如：如果Tensor中数据类型是int64，则对应的C++数据类型为int64_t，所以dtype值为typeid(int64_t).name()，
    #      其在MacOS下为'x'，linux下为'l'，Windows下为'__int64'，都表示64位整型变量

