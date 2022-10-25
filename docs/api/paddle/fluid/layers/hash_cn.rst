.. _cn_api_fluid_layers_hash:

hash
-------------------------------

.. py:function::  paddle.fluid.layers.hash(input, hash_size, num_hash=1, name=None)




该OP将输入 hash 成为一个整数，该数的值小于给定的 ``hash_size`` 。**仅支持输入为LoDTensor**。

该OP使用的哈希算法是：xxHash - `Extremely fast hash algorithm <https://github.com/Cyan4973/xxHash/tree/v0.6.5>`_ 


参数
::::::::::::

  - **input** (Variable) - 输入是一个 **二维** ``LoDTensor`` 。**输入维数必须为2**。数据类型为：int32、int64。**仅支持LoDTensor**。
  - **hash_size** (int) - 哈希算法的空间大小。输出值将保持在 :math:`[0, hash\_size)` 范围内。
  - **num_hash** (int) - 哈希次数。默认值为1。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
``LoDTensor``

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.hash