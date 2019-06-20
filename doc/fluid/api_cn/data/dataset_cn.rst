#################
dataset
#################


.. _cn_api_paddle_dataset_mnist:

mnist
-------------------------------

MNIST数据集。

此模块将从 http://yann.lecun.com/exdb/mnist/ 下载数据集，并将训练集和测试集解析为paddle reader creator。



.. py:function:: paddle.dataset.mnist.train()

MNIST训练数据集的creator。

它返回一个reader creator, reader中的每个样本的图像像素范围是[-1，1]，标签范围是[0，9]。

返回： 训练数据的reader creator

返回类型：callable



.. py:function:: paddle.dataset.mnist.test()

MNIST测试数据集的creator。

它返回一个reader creator, reader中的每个样本的图像像素范围是[-1，1]，标签范围是[0，9]。

返回： 测试数据集的reader creator

返回类型：callable



.. py:function:: paddle.dataset.mnist.convert(path)

将数据集转换为recordio格式。



.. _cn_api_paddle_dataset_cifar:

cifar
-------------------------------

CIFAR数据集。

此模块将从 https://www.cs.toronto.edu/~kriz/cifar.html 下载数据集，并将训练集和测试集解析为paddle reader creator。

cifar-10数据集由10个类别的60000张32x32彩色图像组成，每个类别6000张图像。共有5万张训练图像，1万张测试图像。

cifar-100数据集与cifar-10类似，只是它有100个类，每个类包含600张图像。每个类有500张训练图像和100张测试图像。



.. py:function:: paddle.dataset.cifar.train100()

CIFAR-100训练数据集的creator。

它返回一个reader creator, reader中的每个样本的图像像素范围是[0，1]，标签范围是[0，9]。

返回： 训练数据集的reader creator。

返回类型：callable


.. py:function:: paddle.dataset.cifar.test100()

CIFAR-100测试数据集的creator。

它返回一个reader creator, reader中的每个样本的图像像素范围是[0，1]，标签范围是[0，9]。

返回： 测试数据集的reader creator

返回类型：callable


.. py:function:: paddle.dataset.cifar.train10(cycle=False)

CIFAR-10训练数据集的creator。

它返回一个reader creator, reader中的每个样本的图像像素范围是[0，1]，标签范围是[0，9]。

参数：
    - **cycle** (bool) – 是否循环使用数据集

返回： 训练数据集的reader creator

返回类型：callable


.. py:function:: paddle.dataset.cifar.test10(cycle=False)

CIFAR-10测试数据集的creator。

它返回一个reader creator, reader中的每个样本的图像像素范围是[0，1]，标签范围是[0，9]。

参数：
    - **cycle** (bool) – 是否循环使用数据集

返回： 测试数据集的reader creator

返回类型：callable


.. py:function:: paddle.dataset.cifar.convert(path)

将数据集转换为recordio格式。



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



.. _cn_api_paddle_dataset_imdb:

imdb
-------------------------------

IMDB数据集。

本模块的数据集从 http://ai.stanford.edu/%7Eamaas/data/sentiment/IMDB 数据集。这个数据集包含了25000条训练用电影评论数据，25000条测试用评论数据，且这些评论带有明显情感倾向。此外，该模块还提供了用于构建词典的API。


.. py:function:: paddle.dataset.imdb.build_dict(pattern, cutoff)

从语料库构建一个单词字典，词典的键是word，值是这些单词从0开始的ID。


.. py:function:: paddle.dataset.imdb.train(word_idx)

IMDB训练数据集的creator。


它返回一个reader creator, reader中的每个样本的是一个从0开始的ID序列，标签范围是[0，1]。


参数：
    - **word_idx** (dict) – 词典

返回： 训练数据集的reader creator

返回类型：callable


.. py:function:: paddle.dataset.imdb.test(word_idx)

IMDB测试数据集的creator。

它返回一个reader creator, reader中的每个样本的是一个从0开始的ID序列，标签范围是[0，1]。

参数：
    - **word_idx** (dict) – 词典

返回： 训练数据集的reader creator

返回类型：callable


.. py:function:: paddle.dataset.imdb.convert(path)

将数据集转换为recordio格式。


.. _cn_api_paddle_dataset_imikolov:

imikolov
-------------------------------

imikolov的简化版数据集。

此模块将从 http://www.fit.vutbr.cz/~imikolov/rnnlm/ 下载数据集，并将训练集和测试集解析为paddle reader creator。

.. py:function:: paddle.dataset.imikolov.build_dict(min_word_freq=50)

从语料库构建一个单词字典，字典的键是word，值是这些单词从0开始的ID。

.. py:function:: paddle.dataset.imikolov.train(word_idx, n, data_type=1)

imikolov训练数据集的creator。

它返回一个reader creator, reader中的每个样本的是一个单词ID元组。

参数：
    - **word_idx** (dict) – 词典
    - **n** (int) – 如果类型是ngram，表示滑窗大小；否则表示序列最大长度
    - **data_type** (数据类型的成员变量(NGRAM 或 SEQ)) – 数据类型 (ngram 或 sequence)

返回： 训练数据集的reader creator

返回类型：callable

.. py:function::paddle.dataset.imikolov.test(word_idx, n, data_type=1)

imikolov测试数据集的creator。

它返回一个reader creator, reader中的每个样本的是一个单词ID元组。

参数：
    - **word_idx** (dict) – 词典
    - **n** (int) – 如果类型是ngram，表示滑窗大小；否则表示序列最大长度
    - **data_type** (数据类型的成员变量(NGRAM 或 SEQ)) – 数据类型 (ngram 或 sequence)

返回： 测试数据集的reader creator

返回类型：callable


.. py:function:: paddle.dataset.imikolov.convert(path)

将数据集转换为recordio格式。



.. _cn_api_paddle_dataset_movielens:

movielens
-------------------------------


Movielens 1-M数据集。

Movielens 1-M数据集是由GroupLens Research采集的6000个用户对4000个电影的的100万个评级。 该模块将从 http://files.grouplens.org/datasets/movielens/ml-1m.zip 下载Movielens 1-M数据集，并将训练集和测试集解析为paddle reader creator。


.. py:function:: paddle.dataset.movielens.get_movie_title_dict()

获取电影标题词典。

.. py:function:: paddle.dataset.movielens.max_movie_id()

获取电影ID的最大值。


.. py:function:: paddle.dataset.movielens.max_user_id()

获取用户ID的最大值。


.. py:function:: paddle.dataset.movielens.max_job_id()

获取职业ID的最大值。


.. py:function:: paddle.dataset.movielens.movie_categories()

获取电影类别词典。

.. py:function:: paddle.dataset.movielens.user_info()

获取用户信息词典。

.. py:function:: paddle.dataset.movielens.movie_info()

获取电影信息词典。

.. py:function:: paddle.dataset.movielens.convert(path)

将数据集转换为recordio格式。

.. py:class:: paddle.dataset.movielens.MovieInfo(index, categories, title)

电影ID，标题和类别信息存储在MovieInfo中。


.. py:class:: paddle.dataset.movielens.UserInfo(index, gender, age, job_id)

用户ID，性别，年龄和工作信息存储在UserInfo中。



.. _cn_api_paddle_dataset_sentiment:

sentiment
-------------------------------

脚本获取并预处理由NLTK提供的movie_reviews数据集。


.. py:function:: paddle.dataset.sentiment.get_word_dict()

按照样本中出现的单词的频率对单词进行排序。

返回： words_freq_sorted

.. py:function:: paddle.dataset.sentiment.train()

默认的训练集reader creator。

.. py:function:: paddle.dataset.sentiment.test()

默认的测试集reader creator。

.. py:function:: paddle.dataset.sentiment.convert(path)

将数据集转换为recordio格式。



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






.. _cn_api_paddle_dataset_wmt16:

wmt16
-------------------------------

ACL2016多模式机器翻译。 有关更多详细信息，请访问此网站：http://www.statmt.org/wmt16/multimodal-task.html#task1

如果您任务中使用该数据集，请引用以下文章：Multi30K：多语言英语 - 德语图像描述。

@article{elliott-EtAl:2016:VL16, author = {{Elliott}, D. and {Frank}, S. and {Sima”an}, K. and {Specia}, L.}, title = {Multi30K: Multilingual English-German Image Descriptions}, booktitle = {Proceedings of the 6th Workshop on Vision and Language}, year = {2016}, pages = {70–74}, year = 2016
}

.. py:function:: paddle.dataset.wmt16.train(src_dict_size, trg_dict_size, src_lang='en')

WMT16训练集reader（读取器）。

此功能返回可读取训练数据的reader。 reader返回的每个样本由三个字段组成：源语言单词索引序列，目标语言单词索引序列和下一单词索引序列。

注意：训练数据的原始内容如下： http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz

paddle.dataset.wmt16使用moses的tokenization脚本提供原始数据集的tokenized版本： https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl

参数：
    - **src_dict_size** (int) – 源语言词典的大小。三个特殊标记将被添加到所述词典：<S>为起始标记，<E>为结束标记，<UNK>为未知单词。
    - **trg_dict_size**  (int) – 目标语言字典的大小。三个特殊标记将被添加到所述词典：<S>为起始标记，<E>为结束标记，<UNK>为未知单词。
    - **src_lang**  (string) – 一个字符串，指示哪种语言是源语言。 可用选项包括：英语为“en”，德国为“de”。

返回: 读训练集数据的reader

返回类型: callable



.. py:function:: paddle.dataset.wmt16.test(src_dict_size, trg_dict_size, src_lang='en')


WMT16测试(test)集reader。

此功能返回可读取测试数据的reader。reader返回的每个样本由三个字段组成：源语言单词索引序列，目标语言单词索引序列和下一单词索引序列。

注意：原始测试数据如下： http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz

paddle.dataset.wmt16使用moses的tokenization脚本提供原始数据集的tokenized版本： https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl


参数：
    - **src_dict_size** (int) – 源语言词典的大小。三个特殊token将被添加到所述词典：<S>为起始标记，<E>为结束标记，<UNK>为未知单词。
    - **trg_dict_size**  (int) – 目标语言字典的大小。三个特殊token将被添加到所述词典：<S>为起始标记，<E>为结束标记，<UNK>为未知单词。
    - **src_lang**  (string) – 一个字符串，指示哪种语言是源语言。 可用选项包括：英语为“en”，德国为“de”。


返回: 读测试集数据的reader

返回类型: callable


.. py:function:: paddle.dataset.wmt16.validation(src_dict_size, trg_dict_size, src_lang='en')

WMT16验证(validation)集reader。

此功能返回可读取验证数据的reader 。reader返回的每个样本由三个字段组成：源语言单词索引序列，目标语言单词索引序列和下一单词索引序列。

注意：验证数据的原始内容如下：http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz

paddle.dataset.wmt16使用moses的tokenization脚本提供原始数据集的tokenized版本：https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl



参数：
    - **src_dict_size** (int) – 源语言词典的大小。三个特殊token将被添加到所述词典：<S>为起始标记，<E>为结束标记，<UNK>为未知单词。
    - **trg_dict_size**  (int) – 目标语言字典的大小。三个特殊token将被添加到所述词典：<S>为起始标记，<E>为结束标记，<UNK>为未知单词。
    - **src_lang**  (string) – 一个字符串，指示哪种语言是源语言。 可用选项包括：英语为“en”，德国为“de”。


返回: 读集数据的reader

返回类型: callable







.. py:function:: paddle.dataset.wmt16.get_dict(lang, dict_size, reverse=False)


返回指定语言的词典(word dictionary)。


参数：
    - **lang** （string） - 表示哪种语言是源语言的字符串。 可用选项包括：英语为“en”，德国为“de”。
    - **dict_size** （int） - 指定语言字典的大小。
    - **reverse** （bool） - 如果reverse设置为False，则返回的python字典将使用word作为键并使用index作为值。 如果reverse设置为True，则返回的python字典将使用index作为键，将word作为值。

返回：特定语言的单词词典。

返回类型： dict




.. py:function:: paddle.dataset.wmt16.fetch()

下载完整的数据集。


.. py:function:: paddle.dataset.wmt16.convert(path, src_dict_size, trg_dict_size, src_lang)


将数据集转换为recordio格式。



