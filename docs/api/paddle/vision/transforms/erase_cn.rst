.. _cn_api_vision_transforms_erase:

erase
-------------------------------

.. py:function:: paddle.vision.transforms.erase(img, i, j, h, w, v, inplace=False)

使用给定的值擦除输入图像中的选定区域中的像素。

参数
:::::::::

    - **img** (paddle.Tensor|np.array|PIL.Image) - 输入的图像。对于 Tensor 类型的输入，形状应该为(C, H, W)。对于 np.array 类型的输入，形状应该为(H, W, C)。
    - **i** (int) - 擦除区域左上角点的纵坐标。
    - **j** (int) - 擦除区域左上角点的横坐标。
    - **h** (int) - 擦除区域的高。
    - **w** (int) - 擦除区域的宽。
    - **v** (paddle.Tensor|np.array) - 用于替换擦除区域中像素的值。当输入为 np.array 或者 PIL.Image 类型时，需要为 np.array 类型。
    - **inplace** (bool，可选) - 该变换是否在原地操作。默认值：False。

返回
:::::::::

    ``paddle.Tensor`` 或 ``numpy.array`` 或 ``PIL.Image``，擦除后的图像，类型与输入图像的类型一致。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.erase
