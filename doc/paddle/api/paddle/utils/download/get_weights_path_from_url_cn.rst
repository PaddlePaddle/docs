.. _cn_api_paddle_utils_download_get_weights_path_from_url:

get_weights_path_from_url
-------------------------------

.. py:function:: paddle.utils.download.get_weights_path_from_url(url, md5sum=None)

 从 ``WEIGHT_HOME`` 文件夹获取权重，如果不存在，就从url下载

参数：
  - **url** (str) - 下载的链接。
  - **md5sum** (str，可选) - 下载文件的md5值。默认值：None。

返回：权重的本地路径。


**代码示例**：

.. code-block:: python

    from paddle.utils.download import get_weights_path_from_url

    resnet18_pretrained_weight_url = 'https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams'
    local_weight_path = get_weights_path_from_url(resnet18_pretrained_weight_url)
