.. _cn_api_paddle_dataset_wmt14:

wmt14
-------------------------------

WMT14数据集。 原始WMT14数据集太大，所以提供了一组小数据集。 该模块将从 http://paddlepaddle.cdn.bcebos.com/demo/wmt_shrinked_data/wmt14.tgz 下载数据集，并将训练集和测试集解析为paddle reader creator。


.. py:function:: paddle.dataset.wmt14.train(dict_size)

WMT14训练集creator。

它返回一个reader creator，reader中的每个样本都是源语言单词ID序列，目标语言单词ID序列和下一个单词ID序列。

返回：训练集reader creator

返回类型：callable



.. py:function:: paddle.dataset.wmt14.test(dict_size)


WMT14测试集creator。

它返回一个reader creator，reader中的每个样本都是源语言单词ID序列，目标语言单词ID序列和下一个单词ID序列。

返回：测试集reader creator

返回类型：callable




.. py:function:: paddle.dataset.wmt14.convert(path)

将数据集转换为recordio格式。






