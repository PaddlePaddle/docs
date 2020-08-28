.. _cn_api_tensor_random_randperm:

randperm
-------------------------------

.. py:function:: paddle.randperm(n, dtype="int64", name=None)

:alias_main: paddle.randperm
:alias: paddle.tensor.randperm, paddle.tensor.random.randperm

该OP返回一个数值在0到n-1、随机排列的1-D Tensor，数据类型为 ``dtype``。

参数:
::::::::::
  - **n** (int) - 随机序列的上限（不包括在序列中），应该大于0。 
  - **dtype** (str|np.dtype|core.VarDesc.VarType, 可选) - 输出Tensor的数据类型，支持int32、int64、float32、float64。默认值为"int64".
  - **name** (str, 可选) - 输出的名字。一般无需设置，默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回
::::::::::
  Tensor：一个数值在0到n-1、随机排列的1-D Tensor，数据类型为 ``dtype`` 。

抛出异常
::::::::::
  - ValueError - 如果 ``n`` 不大于0.
  - TypeError - 如果 ``dtype`` 不是int32、int64、float32、float64.

代码示例
::::::::::

..  code-block:: python

    import paddle

    paddle.enable_imperative()

    result_1 = paddle.randperm(5)
    # [4 1 2 3 0]

    result_2 = paddle.randperm(7, 'int32')
    # [1 6 2 0 4 3 5]   
