.. _cn_api_paddle_vision_transforms_rotate:

rotate
-------------------------------

.. py:function:: paddle.vision.transforms.rotate(img, angle, interpolation="nearest", expand=False, center=None, fill=0)

按角度旋转图像。

参数
:::::::::

    - **img** (PIL.Image|numpy.ndarray|paddle.Tensor) - 输入图像。
    - **angle** (float|int) - 逆时针旋转角度。
    - **interpolation** (str，可选) - 插值方法。若省略或图像只有一个通道，则根据后端使用 PIL.Image.NEAREST 或 cv2.INTER_NEAREST。PIL 后端支持 ``"nearest"``、``"bilinear"``、``"bicubic"``，分别对应 Image.NEAREST、Image.BILINEAR、Image.BICUBIC；cv2 后端分别对应 cv2.INTER_NEAREST、cv2.INTER_LINEAR、cv2.INTER_CUBIC。默认值为 ``"nearest"``。
    - **expand** (bool，可选) - 是否要对旋转后的图片进行大小扩展，默认值：False，不进行扩展。当参数值为 True 时，会对图像大小进行扩展，让其能够足以容纳整个旋转后的图像。当参数值为 False 时，会按照原图像大小保留旋转后的图像。**这个扩展操作的前提是围绕中心旋转且没有平移。**
    - **center** (list|tuple|None，可选) - 旋转的中心点坐标，原点是图片左上角，默认值是图像的中心点。
    - **fill** (list|tuple|int，可选) - 旋转图像外部区域的 RGB 像素填充值。如果为 int 类型，则分别用于所有通道。默认值：0。

返回
:::::::::

    ``PIL.Image``、numpy ndarray 或 paddle.Tensor，旋转后的图像。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.rotate
