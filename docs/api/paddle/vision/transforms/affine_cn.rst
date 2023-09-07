.. _cn_api_paddle_vision_transforms_affine:

affine
-------------------------------

.. py:function:: paddle.vision.transforms.affine(img, angle, translate, scale, shear, interpolation="nearest", fill=0, center=None)

对图像进行仿射变换。

参数
::::::::::::
    - img (PIL.Image|np.array|paddle.Tensor) - 输入图像。
    - angle (float|int) - 旋转角度，顺时针方向。
    - translate (list[float]) - 随机水平平移和垂直平移变化的位移大小。
    - scale (float) - 伸缩变换的比例大小，注意必须大于 0 。
    - shear (list|tuple) - 剪切角度值，顺时针方向，第一个值是平行于 x 轴的剪切量，而第二个值是于平行于 y 轴的剪切量。
    - interpolation (str，可选) - 插值的方法。
        如果这个参数没有设定或者输入图像为单通道，则该参数会根据使用的后端，被设置为 ``PIL.Image.NEAREST`` 或者 ``cv2.INTER_NEAREST`` 。
        当使用 ``pil`` 作为后端时, 支持的插值方法如下:
            - "nearest": Image.NEAREST
            - "bilinear": Image.BILINEAR
            - "bicubic": Image.BICUBIC
        当使用 ``cv2`` 作为后端时, 支持的插值方法如下:
            - "nearest": cv2.INTER_NEAREST
            - "bilinear": cv2.INTER_LINEAR
            - "bicubic": cv2.INTER_CUBIC
    - fill (int|list|tuple，可选) - 对图像扩展时填充的像素值，默认值： 0 ，如果只设定一个数字则所有通道上像素值均为该值。
    - center (2-tuple，可选) - 仿射变换的中心点坐标，原点是图片左上角，默认值是图像的中心点。

返回
::::::::::::

    ``PIL.Image / numpy ndarray / paddle.Tensor`` ，仿射变换后的图像。

代码示例
::::::::::::

COPY-FROM: paddle.vision.transforms.affine
