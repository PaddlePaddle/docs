.. _cn_api_vision_transforms_BatchCompose:

BatchCompose
-------------------------------

.. py:class:: paddle.vision.transforms.BatchCompose(transforms=[])

用于处理批数据的预处理接口组合。

参数
:::::::::

    - transforms (list): - 用于组合的数据预处理接口实例。这些预处理接口所处理的是一批数据。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from paddle.io import DataLoader

    from paddle import set_device
    from paddle.vision.datasets import Flowers
    from paddle.vision.transforms import Compose, BatchCompose, Resize

    class NormalizeBatch(object):
        def __init__(self,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    scale=True,
                    channel_first=True):

            self.mean = mean
            self.std = std
            self.scale = scale
            self.channel_first = channel_first
            if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                    isinstance(self.scale, bool)):
                raise TypeError("{}: input type is invalid.".format(self))
            from functools import reduce
            if reduce(lambda x, y: x * y, self.std) == 0:
                raise ValueError('{}: std is invalid!'.format(self))

        def __call__(self, samples):
            for i in range(len(samples)):
                samples[i] = list(samples[i])
                im = samples[i][0]
                im = im.astype(np.float32, copy=False)
                mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                std = np.array(self.std)[np.newaxis, np.newaxis, :]
                if self.scale:
                    im = im / 255.0
                im -= mean
                im /= std
                if self.channel_first:
                    im = im.transpose((2, 0, 1))
                samples[i][0] = im
            return samples

    transform = Compose([Resize((500, 500))])
    flowers_dataset = Flowers(mode='test', transform=transform)

    device = set_device('cpu')

    collate_fn = BatchCompose([NormalizeBatch()])
    loader = DataLoader(
                flowers_dataset,
                batch_size=4,
                places=device,
                return_list=True,
                collate_fn=collate_fn)

    for data in loader:
        # do something
        break