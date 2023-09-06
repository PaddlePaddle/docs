.. _cn_api_vision_transforms_pad:

pad
-------------------------------

.. py:function:: paddle.vision.transforms.pad(img, padding, fill=0, padding_mode='constant')

使用特定的模式和值来对输入图像进行填充。

参数
:::::::::

    - **img** (PIL.Image|np.ndarray) - 被填充的图像。
    - **padding** (int|list|tuple) - 在图像边界上进行填充的范围。如果提供的是单个 int 值，则该值用于填充图像所有边；如果提供的是长度为 2 的元组/列表，则分别为图像左/右和顶部/底部进行填充；如果提供的是长度为 4 的元组/列表，则按照左，上，右和下的顺序为图像填充。
    - **fill** (int|tuple，可选) - 用于填充的像素值。仅当 padding_mode 为 constant 时参数值有效。默认值：0。如果参数值是一个长度为 3 的元组，则会分别用于填充 R，G，B 通道。
    - **padding_mode** (string，可选) - 填充模式。支持：constant, edge, reflect 或 symmetric。默认值：constant。

        - ``constant`` 表示使用常量值进行填充，该值由 fill 参数指定；
        - ``edge`` 表示使用图像边缘像素值进行填充；
        - ``reflect`` 表示使用原图像的镜像值进行填充（不使用边缘上的值）；比如：使用该模式对 ``[1, 2, 3, 4]`` 的两端分别填充 2 个值，结果是 ``[3, 2, 1, 2, 3, 4, 3, 2]``。
        - ``symmetric`` 表示使用原图像的镜像值进行填充（使用边缘上的值）；比如：使用该模式对 ``[1, 2, 3, 4]`` 的两端分别填充 2 个值，结果是 ``[2, 1, 1, 2, 3, 4, 4, 3]``。

返回
:::::::::

``PIL.Image`` 或 ``numpy.ndarray``，填充后的图像。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.pad
