.. _cn_api_vision_transforms_erase:

erase
-------------------------------

.. py:function:: paddle.vision.transforms.erase(img, i, j, h, w, v, inplace=False)

使用给定的值擦除输入图像中的选定区域中的像素。

参数
:::::::::

    - img (paddle.Tensor|np.array|PIL.Image) - 输入的图像。对于Tensor类型的输入，形状应该为(C，H，W)。对于np.array类型的输入，形状应该为(H，W，C)。
    - i (int) - 擦除区域左上角点的纵坐标。
    - j (int) - 擦除区域左上角点的横坐标。
    - h (int) - 擦除区域的高。
    - w (int) - 擦除区域的宽。
    - v (paddle.Tensor|np.array) - 用于替换擦除区域中像素的值。当输入为np.array或者PIL.Image类型时，需要为np.array类型。
    - inplace (bool，可选) - 该变换是否在原地操作。默认值：False。

返回
:::::::::

    ``paddle.Tensor`` 或 ``numpy.array`` 或 ``PIL.Image``，擦除后的图像，类型与输入图像的类型一致。

代码示例
:::::::::

.. code-block:: python

    import paddle

    fake_img = paddle.randn((3, 2, 4)).astype(paddle.float32)
    print(fake_img)

    #Tensor(shape=[3, 2, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
    #       [[[ 0.02169025, -0.97859967, -1.39175487, -1.07478464],
    #         [ 0.20654772,  1.74624777,  0.32268861, -0.13857445]],
    #
    #        [[-0.14993843,  1.10793507, -0.40056887, -1.94395220],
    #         [ 0.41686651,  0.44551995, -0.09356714, -0.60898107]],
    #
    #        [[-0.24998808, -1.47699273, -0.88838995,  0.42629015],
    #         [ 0.56948012, -0.96200180,  0.53355658,  3.20450878]]])

    values = paddle.zeros((1,1,1), dtype=paddle.float32)
    result = paddle.vision.transforms.erase(fake_img, 0, 1, 1, 2, values)

    print(result)

    #Tensor(shape=[3, 2, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
    #       [[[ 0.02169025,  0.        ,  0.        , -1.07478464],
    #         [ 0.20654772,  1.74624777,  0.32268861, -0.13857445]],
    #
    #         [[-0.14993843,  0.        ,  0.        , -1.94395220],
    #           [ 0.41686651,  0.44551995, -0.09356714, -0.60898107]],
    #
    #         [[-0.24998808,  0.        ,  0.        ,  0.42629015],
    #          [ 0.56948012, -0.96200180,  0.53355658,  3.20450878]]])

