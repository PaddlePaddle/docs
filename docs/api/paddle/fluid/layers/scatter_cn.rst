.. _cn_api_fluid_layers_scatter:

scatter
-------------------------------

.. py:function:: paddle.fluid.layers.scatter(input, index, updates, name=None, overwrite=True)




该OP根据index中的索引值将updates数据更新到input中。

.. code-block:: python

  输入：
    input = np.array([[1, 1], [2, 2], [3, 3]])
    index = np.array([2, 1, 0, 1])
    # updates的维度需要和input一样
    # updates 维度 > 1 的shape要和input一样
    updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    overwrite = False
  
  计算过程：
    if not overwrite:
       for i in range(len(index)):
         input[index[i]] = np.zeros((2))

    # 根据index中的索引值取updates中的数据更新到input中去 
    for i in range(len(index)):
      if (overwirte):
        input[index[i]] = updates[i]
      else:
        input[index[i]] += updates[i]

  输出：
    out # np.array([[3, 3], [6, 6], [1, 1]])
    out.shape # [3, 2]

参数
::::::::::::

  - **input** （Variable） - 支持任意纬度的Tensor。支持的数据类型为float32。
  - **index** （Variable） - 表示索引，仅支持1-D Tensor。支持的数据类型为int32，int64。
  - **updates** （Variable） - 根据索引的值将updates Tensor中的对应值更新到input Tensor中，updates Tensor的维度需要和input tensor保持一致，且除了第一维外的其他的维度的大小需要和input Tensor保持相同。支持的数据类型为float32。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **overwrite** （bool，可选） - 如果index中的索引值有重复且overwrite 为True，旧更新值将被新的更新值覆盖；如果为False，新的更新值将同旧的更新值相加。默认值为True。

返回
::::::::::::
返回类型为Variable(Tensor|LoDTensor)，数据类型以及shape大小同输入一致。

代码示例
::::::::::::

..  code-block:: python

    import numpy as np
    import paddle.fluid as fluid

    input = fluid.layers.data(name='data', shape=[3, 2], dtype='float32', append_batch_size=False)
    index = fluid.layers.data(name='index', shape=[4], dtype='int64', append_batch_size=False)
    updates = fluid.layers.data(name='update', shape=[4, 2], dtype='float32', append_batch_size=False)

    output = fluid.layers.scatter(input, index, updates, overwrite=False)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    in_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
    index_data = np.array([2, 1, 0, 1]).astype(np.int64)
    update_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(np.float32)

    res = exe.run(fluid.default_main_program(), feed={'data':in_data, "index":index_data, "update":update_data}, fetch_list=[output])
    print(res)
    # [array([[3., 3.],
    #   [6., 6.],
    #   [1., 1.]], dtype=float32)]




