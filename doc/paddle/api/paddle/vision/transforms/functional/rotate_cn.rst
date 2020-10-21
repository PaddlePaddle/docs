.. _cn_api_vision_transforms_rotate:

rotate
-------------------------------

.. py:function:: paddle.vision.transforms.rotate(img, angle, interpolation=cv2.INTER_LINEAR, expand=False, center=None)

按角度旋转图像。

参数
:::::::::

    - img (numpy.ndarray) - 输入图像。
    - angle (float|int) - 旋转角度，顺时针。
    - interpolation (int，可选) - 调整图片大小时使用的插值模式。默认值: cv2.INTER_LINEAR。
    - expand (bool，可选) - 是否要对旋转后的图片进行大小扩展，默认值: False，不进行扩展。
            当参数值为True时，会对图像大小进行扩展，让其能够足以容纳整个旋转后的图像。
            当参数值为False时，会按照原图像大小保留旋转后的图像。
            **这个扩展操作的前提是围绕中心旋转且没有平移。**
    - center (2-tuple，可选) - 旋转的中心点坐标，原点是图片左上角，默认值是图像的中心点。

返回
:::::::::

    ``numpy ndarray``，旋转后的图像。

代码示例
:::::::::
    
.. code-block:: python
        
    import numpy as np
    from paddle.vision.transforms.functional import rotate


    fake_img = np.random.rand(3, 3, 3).astype('float32')
    print('before rotate:')
    print(fake_img)
    fake_img = rotate(fake_img, 90)
    print('after rotate:')
    print(fake_img)
    """
    before rotate:
    [[[0.9320921  0.311002   0.22388814]
    [0.9551999  0.10015319 0.7481808 ]
    [0.4619514  0.29591113 0.12210595]]

    [[0.77222216 0.3235876  0.5718483 ]
    [0.8797754  0.35876957 0.9330844 ]
    [0.65897316 0.11888863 0.31214228]]

    [[0.7627513  0.05149421 0.41464522]
    [0.2620253  0.7800404  0.990831  ]
    [0.7814754  0.21640824 0.4333755 ]]]
    
    after rotate:
    [[[0.         0.         0.        ]
    [0.7627513  0.05149421 0.41464522]
    [0.77222216 0.3235876  0.5718483 ]]

    [[0.         0.         0.        ]
    [0.2620253  0.7800404  0.990831  ]
    [0.8797754  0.35876957 0.9330844 ]]

    [[0.         0.         0.        ]
    [0.7814754  0.21640824 0.4333755 ]
    [0.65897316 0.11888863 0.31214228]]]
    """
    