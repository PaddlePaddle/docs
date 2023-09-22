.. _cn_api_paddle_vision_transforms_RandomErasing:

RandomErasing
-------------------------------

.. py:class:: paddle.vision.transforms.RandomErasing(prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False, keys=None)

擦除图像中随机选择的矩形区域内的像素。

参数
:::::::::

    - **prob** (float，可选) - 输入数据被执行擦除操作的概率。默认值：0.5。
    - **scale** (sequence，可选) - 擦除区域面积在输入图像的中占比范围。默认值：(0.02, 0.33)。
    - **ratio** (sequence，可选) - 擦除区域的纵横比范围。默认值：(0.3, 3.3)。
    - **value** (int|float|sequence|str，可选) - 擦除区域中像素将被替换为的值。如果 value 是一个数，所有的像素都将被替换为这个数。如果 value 是长为 3 的序列，R,G,B 通道将被对应地替换。如果 value 是"random"，每个像素会被替换为随机值。默认值：0。
    - **inplace** (bool，可选) - 该变换是否在原地操作。默认值：False。
    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值：None。

形状
:::::::::

    - img (paddle.Tensor|np.array|PIL.Image) - 输入的图像数据。对于 Tensor 类型的输入，形状需要为(C, H, W)。对于 np.array 类型的输入，形状为(H, W, C)。
    - output (paddle.Tensor|np.array|PIL.Image) - 返回随机擦除后的图像数据。

返回
:::::::::

    计算 ``RandomErasing`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.RandomErasing
