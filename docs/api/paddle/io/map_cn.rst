.. _cn_api_io_cn_map:

map
-------------------------------

.. py:class:: paddle.io.map(map_func, *args, **kwargs)

此API用于在GPU DataLoader流水线中划分数据预处理的阶段，每个阶段会通过独立的CUDA流和子线程来执行

参数
::::::::::::

    - **map_func** (callable) - 定义阶段内数据预处理的函数。

返回
::::::::::::
    数据预处理函数的输出


代码示例
::::::::::::

.. code-block:: python

    import os
    import paddle
    from paddle.utils.download import get_path_from_url

    DATASET_HOME = os.path.expanduser("~/.cache/paddle/datasets")
    DATASET_URL = "https://paddlemodels.cdn.bcebos.com/ImageNet_stub.tar"
    DATASET_MD5 = "c7110519124a433901cf005a4a91b607"
    BATCH_SIZE = 100
    NUM_SAMPLES = 100

    data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                  DATASET_MD5)

    def imagenet_pipeline():
        def imagenet_reader(indices):
            return paddle.vision.reader.file_label_loader(
                                data_root, indices, BATCH_SIZE)

        outs = paddle.io.data_reader(imagenet_reader,
                            BATCH_SIZE, NUM_SAMPLES)
        image = outs[:-1]
        label = outs[-1]

        def decode(image):
            image = paddle.vision.ops.image_decode_random_crop(image, num_threads=4)
            return image
        def resize(image):
            image = paddle.vision.ops.image_resize(image, size=224)
            return image

        image = paddle.io.map(decode, image)
        image = paddle.io.map(resize, image)

        return {'image': image, 'label': label}

    # only support GPU version
    if not paddle.get_device() == 'cpu':
        dataloader = paddle.io.DataLoader(imagenet_pipeline)
        for data in dataloader:
            print(data['image'].shape, data['label'].shape)
