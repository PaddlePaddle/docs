.. _cn_api_vision_transforms_erase:

erase
-------------------------------

.. py:function:: paddle.vision.transforms.erase(img, i, j, h, w, v, inplace=False)

使用给定的值擦除输入图像中的选定区域中的像素。

参数
:::::::::

    - img (paddle.Tensor|np.array|PIL.Image) - 输入的图像。对于Tensor类型的输入，形状应该为(C, H, W)。对于np.array类型的输入，形状应该为(H, W, C)。
    - i (int) - 擦除区域左上角点的纵坐标。
    - j (int) - 擦除区域左上角点的横坐标。
    - h (int) - 擦除区域的高。
    - w (int) - 擦除区域的宽。
    - v (paddle.Tensor|np.array) - 用于替换擦除区域中像素的值。当输入为np.array或者PIL.Image类型时，需要为np.array类型。
    - inplace (bool, 可选) - 该变换是否在原地操作。默认值：False。

返回
:::::::::

    ``paddle.Tensor 或 numpy.array 或PIL.Image``，擦除后的图像，类型与输入图像的类型一致。

代码示例
:::::::::

.. code-block:: python

    import paddle
                
    fake_img = paddle.randn((3, 10, 10)).astype(paddle.float32)
    values = paddle.zeros((1,1,1), dtype=paddle.float32)
    result = paddle.vision.transforms.erase(fake_img, 4, 4, 3, 3, values)       
