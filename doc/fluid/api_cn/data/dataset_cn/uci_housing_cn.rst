.. _cn_api_paddle_dataset_uci_housing:

uci_housing
-------------------------------



UCI Housing数据集。

该模块将从 https://archive.ics.uci.edu/ml/machine-learning-databases/housing/下载数据集，并将训练集和测试集解析为paddle reader creator。



.. py:function:: paddle.dataset.uci_housing.train()

UCI_HOUSING训练集creator。

它返回一个reader creator，reader中的每个样本都是正则化和价格编号后的特征。

返回：训练集reader creator

返回类型：callable



.. py:function:: paddle.dataset.uci_housing.test()


UCI_HOUSING测试集creator。

它返回一个reader creator，reader中的每个样本都是正则化和价格编号后的特征。


返回：测试集reader creator

返回类型：callable






