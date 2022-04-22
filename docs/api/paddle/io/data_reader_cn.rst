.. _cn_api_io_cn_map:

map
-------------------------------

.. py:class:: paddle.io.data_reader(reader_func, batch_size=1, num_samples=1, shuffle=False, drop_last=False, seed=None)

此API用于在GPU DataLoader流水线中启动数据集读取阶段，此阶段会通过独立的子线程来执行

参数
::::::::::::

    - **reader_func** (callable) - 定义阶段内数据集读取的函数。
    - **batch_size** (int) - 每个批次的样本数，默认为1。
    - **num_samples** (int) - 总共读取数据集的样本数，默认为1。
    - **shuffle** (bool) - 是否打乱数据集读取顺序，默认为False。
    - **drop_last** (bool) - 是否丢弃因数据集样本数不能被 ``batch_size`` 整除而产生的最后一个不完整的批次，默认为False。
    - **seed** (int, 可选) - 打乱数据集读取顺序时使用的随机种子，默认为None。

返回
::::::::::::
    数据集读取函数的输出


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
