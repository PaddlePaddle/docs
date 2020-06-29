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



