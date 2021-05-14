.. _cn_api_tensor_expand:

expand
-------------------------------

.. py:function:: paddle.expand(x, shape, name=None)

根据 ``shape`` 指定的形状扩展 ``x`` ，扩展后， ``x`` 的形状和 ``shape`` 指定的形状一致。

``x`` 的维数和 ``shape`` 的元素数应小于等于6，并且 ``shape`` 中的元素数应该大于等于 ``x`` 的维数。扩展的维度的维度值应该为1。

参数
:::::::::
    - x (Tensor) - 输入的Tensor，数据类型为：bool、float32、float64、int32或int64。
    - shape (tuple|list|Tensor) - 给定输入 ``x`` 扩展后的形状,若 ``shape`` 为list或者tuple，则其中的元素值应该为整数或者1-D Tensor，若 ``shape`` 类型为Tensor，则其应该为1-D Tensor。
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
``Tensor`` ，数据类型与 ``x`` 相同。

代码示例
:::::::::

.. code-block:: python

       import paddle
               
       data = paddle.to_tensor([1, 2, 3], dtype='int32')
       out = paddle.expand(data, shape=[2, 3])
       print(out)
       # [[1, 2, 3], [1, 2, 3]]

