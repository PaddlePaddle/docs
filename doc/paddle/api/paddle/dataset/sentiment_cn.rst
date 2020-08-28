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



