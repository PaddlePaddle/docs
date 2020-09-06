.. _cn_api_paddle_dataset_Conll05:

Conll05
-------------------------------

Conll05数据集。Paddle深度学习基础中的语义角色标注文档使用这个数据集为例。因为Conll05数据集不是免费公开的，所以默认下载的url是Conll05的测试集（它是公开的）。用户可以将url和md5更改为其Conll数据集。并采用基于维基百科语料库的预训练词向量模型对SRL模型进行初始化。


.. py:function:: paddle.dataset.conll05.get_dict()

获取维基百科语料库的单词、动词和标签字典。


.. py:function:: paddle.dataset.conll05.get_embedding()

获取基于维基百科语料库的训练词向量。



.. py:function:: paddle.dataset.conll05.test()

Conll05测试数据集的creator。

因为训练数据集不是免费公开的，所以用测试数据集进行训练。它返回一个reader creator，reader中的每个样本都有九个特征，包括句子序列、谓词、谓词上下文、谓词上下文标记和标记序列。

返回： 训练数据集的reader creator

返回类型：callable



