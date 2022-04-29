.. _cn_api_vision_transforms_affine:

affine
-------------------------------

.. py:function:: paddle.vision.transforms.affine(img, angle, translate, scale, shear, interpolation="nearest", fill=0, center=None)

对图像进行仿射变换。

参数
:::::::::

    - img (PIL.Image|np.array|paddle.Tensor) - 输入图像。
    - angle (float|int) - 旋转角度，顺时针方向。
    - translate (list, tuple) - 水平方向和垂直方向的位移量。
    - scale (float|int) - 伸缩比例，注意必须大于0.0。
    - shear (list, tuple) - 剪切角度值，顺时针方向，第一个值是平行于x轴的剪切量，而第二个值是于平行于y轴的剪切量。
    - interpolation (str，可选): 插值方式。如果省略，或者图像只有一个通道，则根据后端将其设置为PIL.Image.NEAREST或cv2.INTER_NEAREST。使采用pil后端时，支持方法如下："nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC;当采用cv2后端时，支持方法如下："nearest": cv2.INTER_NEAREST,  - "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC.
    - fill (int，可选) - 对图像扩展时填充的值。默认值：0。
    - center (2-tuple，可选) - 仿射变换的中心点坐标，原点是图片左上角，默认值是图像的中心点。

返回
:::::::::

    ``PIL.Image / numpy ndarray / paddle.Tensor``，仿射变换后的图像。

代码示例
:::::::::
    
.. code-block:: python
        
    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import functional as F

    fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

    fake_img = Image.fromarray(fake_img)

    affined_img = F.affine(fake_img, [-90, 90], translate=[0.2, 0.2], scale=0.5, shear=[-10, 10])
    print(affined_img.size)
    
