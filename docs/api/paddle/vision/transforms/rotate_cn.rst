.. _cn_api_paddle_vision_transforms_rotate:

rotate
-------------------------------

.. py:function:: paddle.vision.transforms.rotate(img, angle, resample=False, expand=False, center=None, fill=0)

按角度旋转图像。

参数
:::::::::

    - **img** (PIL.Image|numpy.ndarray) - 输入图像。
    - **angle** (float|int) - 旋转角度，顺时针。
    - **resample** (int|str，可选) - 可选的重采样滤波器。如果省略，或者图像只有一个通道，则根据后端将其设置为 PIL.Image.NEAREST 或 cv2.INTER_NEAREST。使采用 pil 后端时，支持方法如下：- "nearest": Image.NEAREST, -"bilinear": Image.BILINEAR, -"bicubic": Image.BICUBIC；当采用 cv2 后端时，支持方法如下：- "nearest": cv2.INTER_NEAREST,  - "bilinear": cv2.INTER_LINEAR, - "bicubic": cv2.INTER_CUBIC。
    - **expand** (bool，可选) - 是否要对旋转后的图片进行大小扩展，默认值：False，不进行扩展。当参数值为 True 时，会对图像大小进行扩展，让其能够足以容纳整个旋转后的图像。当参数值为 False 时，会按照原图像大小保留旋转后的图像。**这个扩展操作的前提是围绕中心旋转且没有平移。**
    - **center** (tuple[int, int]，可选) - 旋转的中心点坐标，原点是图片左上角，默认值是图像的中心点。
    - **fill** (int，可选) - 旋转图像外部区域的 RGB 像素填充值。如果为 int 类型，则分别用于所有通道。默认值：0。

返回
:::::::::

    ``PIL.Image 或 numpy ndarray``，旋转后的图像。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.rotate
