.. _cn_api_tensor_clip:

clip
-------------------------------

.. py:function:: paddle.clip(x, min=None, max=None, name=None)




该OP将输入的所有元素进行剪裁，使得输出元素限制在[min, max]内，具体公式如下：

.. math::

        Out = MIN(MAX(x, min), max) 

参数：
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64。
    - min (float32|Tensor, 可选) - 裁剪的最小值，输入中小于该值的元素将由该元素代替，若参数为空，则不对输入的最小值做限制。数据类型可以是float32或形状为[1]的Tensor，类型可以为int32，float32，float64，默认值为None。
    - max (float32|Tensor, 可选) - 裁剪的最大值，输入中大于该值的元素将由该元素代替，若参数为空，则不对输入的最大值做限制。数据类型可以是float32或形状为[1]的Tensor，类型可以为int32，float32，float64，默认值为None。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：输出Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型：Tensor

**代码示例**：

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    x = np.array([[1.2,3.5], [4.5,6.4]]).astype('float32')
    x1 = paddle.to_tensor(x)
    out1 = paddle.clip(x1, min=3.5, max=5.0)
    out2 = paddle.clip(x1, min=2.5)
    print(out1.numpy())
    # [[3.5, 3.5]
    # [4.5, 5.0]]
    print(out2.numpy())
    # [[2.5, 3.5]
    # [[4.5, 6.4]
