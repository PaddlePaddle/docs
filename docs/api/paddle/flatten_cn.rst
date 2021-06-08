.. _cn_api_fluid_layers_flatten:

flatten
-------------------------------

.. py:function::  paddle.flatten(x, start_axis=0, stop_axis=-1, name=None)




flatten op 根据给定的start_axis 和 stop_axis 将连续的维度展平。

请注意，在动态图模式下，输出Tensor将与输入Tensor共享数据，并且没有Tensor数据拷贝的过程。
如果不希望输入与输出共享数据，请使用 `Tensor.clone` ，例如 `flatten_clone_x = x.flatten().clone()` 。

例如：

.. code-block:: text

    Case 1:

      给定
        X.shape = (3, 100, 100, 4)
      且
        start_axis = 1
        stop_axis = 2

      得到:
        Out.shape = (3, 100 * 100, 4)

    Case 2:

      给定
        X.shape = (3, 100, 100, 4)
      且
        start_axis = 0
        stop_axis = -1

      得到:
        Out.shape = (3 * 100 * 100 * 4)

参数：
  - **x** (Tensor) - 多维Tensor, 数据类型可以为float32，float64，int8，int32或int64。
  - **start_axis** (int) - flatten展开的起始维度。
  - **stop_axis** (int) - flatten展开的结束维度。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回: ``Tensor``, 一个 Tensor，它包含输入Tensor的数据，但维度发生变化。输入将按照给定的start_axis 和 stop_axis展开。数据类型与输入x相同。

返回类型: Tensor

抛出异常：
  - ValueError: 如果 x 不是一个Tensor
  - ValueError: 如果start_axis或者stop_axis不合法

**代码示例**

.. code-block:: python

    import paddle

    image_shape=(2, 3, 4, 4)
    x = paddle.arange(end=image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3])
    img = paddle.reshape(x, image_shape) / 100
    
    out = paddle.flatten(img, start_axis=1, stop_axis=2)
    # out shape is [2, 12, 4]

    # 在动态图模式下，输出out与输入img共享数据
    img[0, 0, 0, 0] = -1
    print(out[0, 0, 0]) # [-1]


