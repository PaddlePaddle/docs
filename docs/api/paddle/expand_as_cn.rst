.. _cn_api_tensor_expand_as:

expand_as
-------------------------------

.. py:function:: paddle.expand_as(x, y, name=None)

根据 ``y`` 的形状扩展 ``x`` ，扩展后， ``x`` 的形状和 ``y`` 的形状相同。

``x`` 的维数和 ``y`` 的维数应小于等于6，并且 ``y`` 的维数应该大于等于 ``x`` 的维数。扩展的维度的维度值应该为1。

参数
:::::::::
    - x (Tensor) - 输入的Tensor，数据类型为：bool、float32、float64、int32或int64。
    - y (Tensor) - 给定输入 ``x`` 扩展后的形状。
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
``Tensor`` ，数据类型与 ``x`` 相同。

代码示例
:::::::::

.. code-block:: python

       import paddle
       import numpy as np
               
       np_data_x = np.array([1, 2, 3]).astype('int32')
       np_data_y = np.array([[1, 2, 3], [4, 5, 6]]).astype('int32')
       data_x = paddle.to_tensor(np_data_x)
       data_y = paddle.to_tensor(np_data_y)
       out = paddle.expand_as(data_x, data_y)
       np_out = out.numpy()
       # [[1, 2, 3], [1, 2, 3]]

