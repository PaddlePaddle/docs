IMDB 数据集使用BOW网络的文本分类
================================

本示例教程演示如何在IMDB数据集上用简单的BOW网络完成文本分类的任务。

IMDB数据集是一个对电影评论标注为正向评论与负向评论的数据集，共有25000条文本数据作为训练集，25000条文本数据作为测试集。
该数据集的官方地址为： http://ai.stanford.edu/~amaas/data/sentiment/

环境设置
--------

本示例基于飞桨开源框架2.0版本。

.. code:: ipython3

    import paddle
    import numpy as np
    
    paddle.disable_static()
    print(paddle.__version__)


.. parsed-literal::

    2.0.0-beta0


加载数据
--------

我们会使用\ ``paddle.dataset``\ 完成数据下载，构建字典和准备数据读取器。在飞桨框架2.0版本中，推荐使用padding的方式来对同一个batch中长度不一的数据进行补齐，所以在字典中，我们还会添加一个特殊的\ ``<pad>``\ 词，用来在后续对batch中较短的句子进行填充。

.. code:: ipython3

    print("Loading IMDB word dict....")
    word_dict = paddle.dataset.imdb.word_dict()
    
    train_reader = paddle.dataset.imdb.train(word_dict)
    test_reader = paddle.dataset.imdb.test(word_dict)



.. parsed-literal::

    Loading IMDB word dict....


.. code:: ipython3

    # add a pad token to the dict for later padding the sequence
    word_dict['<pad>'] = len(word_dict)
    
    for k in list(word_dict)[:5]:
        print("{}:{}".format(k.decode('ASCII'), word_dict[k]))
    
    print("...")
    
    for k in list(word_dict)[-5:]:
        print("{}:{}".format(k if isinstance(k, str) else k.decode('ASCII'), word_dict[k]))
    
    print("totally {} words".format(len(word_dict)))


.. parsed-literal::

    the:0
    and:1
    a:2
    of:3
    to:4
    ...
    virtual:5143
    warriors:5144
    widely:5145
    <unk>:5146
    <pad>:5147
    totally 5148 words


参数设置
--------

在这里我们设置一下词表大小，\ ``embedding``\ 的大小，batch_size，等等

.. code:: ipython3

    vocab_size = len(word_dict)
    emb_size = 256
    seq_len = 200
    batch_size = 32
    epoch_num = 2
    pad_id = word_dict['<pad>']
    
    classes = ['negative', 'positive']
    
    def ids_to_str(ids):
        #print(ids)
        words = []
        for k in ids:
            w = list(word_dict)[k]
            words.append(w if isinstance(w, str) else w.decode('ASCII'))
        return " ".join(words)

在这里，取出一条数据打印出来看看，可以对数据有一个初步直观的印象。

.. code:: ipython3

    # 取出来第一条数据看看样子。
    sent, label = next(train_reader())
    print(sent, label)
    
    print(ids_to_str(sent))
    print(classes[label])


.. parsed-literal::

    [5146, 43, 71, 6, 1092, 14, 0, 878, 130, 151, 5146, 18, 281, 747, 0, 5146, 3, 5146, 2165, 37, 5146, 46, 5, 71, 4089, 377, 162, 46, 5, 32, 1287, 300, 35, 203, 2136, 565, 14, 2, 253, 26, 146, 61, 372, 1, 615, 5146, 5, 30, 0, 50, 3290, 6, 2148, 14, 0, 5146, 11, 17, 451, 24, 4, 127, 10, 0, 878, 130, 43, 2, 50, 5146, 751, 5146, 5, 2, 221, 3727, 6, 9, 1167, 373, 9, 5, 5146, 7, 5, 1343, 13, 2, 5146, 1, 250, 7, 98, 4270, 56, 2316, 0, 928, 11, 11, 9, 16, 5, 5146, 5146, 6, 50, 69, 27, 280, 27, 108, 1045, 0, 2633, 4177, 3180, 17, 1675, 1, 2571] 0
    <unk> has much in common with the third man another <unk> film set among the <unk> of <unk> europe like <unk> there is much inventive camera work there is an innocent american who gets emotionally involved with a woman he doesnt really understand and whose <unk> is all the more striking in contrast with the <unk> br but id have to say that the third man has a more <unk> storyline <unk> is a bit disjointed in this respect perhaps this is <unk> it is presented as a <unk> and making it too coherent would spoil the effect br br this movie is <unk> <unk> in more than one sense one never sees the sun shine grim but intriguing and frightening
    negative


用padding的方式对齐数据
-----------------------

文本数据中，每一句话的长度都是不一样的，为了方便后续的神经网络的计算，常见的处理方式是把数据集中的数据都统一成同样长度的数据。这包括：对于较长的数据进行截断处理，对于较短的数据用特殊的词\ ``<pad>``\ 进行填充。接下来的代码会对数据集中的数据进行这样的处理。

.. code:: ipython3

    def create_padded_dataset(reader):
        padded_sents = []
        labels = []
        for batch_id, data in enumerate(reader):
            sent, label = data
            padded_sent = sent[:seq_len] + [pad_id] * (seq_len - len(sent))
            padded_sents.append(padded_sent)
            labels.append(label)
        return np.array(padded_sents), np.expand_dims(np.array(labels), axis=1)
    
    train_sents, train_labels = create_padded_dataset(train_reader())
    test_sents, test_labels = create_padded_dataset(test_reader())
    
    print(train_sents.shape)
    print(train_labels.shape)
    print(test_sents.shape)
    print(test_labels.shape)
    
    for sent in train_sents[:3]:
        print(ids_to_str(sent))



.. parsed-literal::

    (25000, 200)
    (25000, 1)
    (25000, 200)
    (25000, 1)
    <unk> has much in common with the third man another <unk> film set among the <unk> of <unk> europe like <unk> there is much inventive camera work there is an innocent american who gets emotionally involved with a woman he doesnt really understand and whose <unk> is all the more striking in contrast with the <unk> br but id have to say that the third man has a more <unk> storyline <unk> is a bit disjointed in this respect perhaps this is <unk> it is presented as a <unk> and making it too coherent would spoil the effect br br this movie is <unk> <unk> in more than one sense one never sees the sun shine grim but intriguing and frightening <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
    <unk> is the most original movie ive seen in years if you like unique thrillers that are influenced by film noir then this is just the right cure for all of those hollywood summer <unk> <unk> the theaters these days von <unk> <unk> like breaking the waves have gotten more <unk> but this is really his best work it is <unk> without being distracting and offers the perfect combination of suspense and dark humor its too bad he decided <unk> cameras were the wave of the future its hard to say who talked him away from the style he <unk> here but its everyones loss that he went into his heavily <unk> <unk> direction instead <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
    <unk> von <unk> is never <unk> in trying out new techniques some of them are very original while others are best <unk> br he depicts <unk> germany as a <unk> train journey with so many cities lying in ruins <unk> <unk> a young american of german descent feels <unk> to help in their <unk> it is not a simple task as he quickly finds outbr br his uncle finds him a job as a night <unk> on the <unk> <unk> line his job is to <unk> to the needs of the passengers when the shoes are <unk> a <unk> mark is made on the <unk> a terrible argument <unk> when a passengers shoes are not <unk> despite the fact they have been <unk> there are many <unk> to the german <unk> of <unk> to such stupid <unk> br the <unk> journey is like an <unk> <unk> mans <unk> through life with all its <unk> and <unk> in one sequence <unk> <unk> through the back <unk> to discover them filled with <unk> bodies appearing to have just escaped from <unk> these images horrible as they are are <unk> as in a dream each with its own terrible impact yet <unk> br


组建网络
--------

本示例中，我们将会使用一个不考虑词的顺序的BOW的网络，在查找到每个词对应的embedding后，简单的取平均，作为一个句子的表示。然后用\ ``Linear``\ 进行线性变换。为了防止过拟合，我们还使用了\ ``Dropout``\ 。

.. code:: ipython3

    class MyNet(paddle.nn.Layer):
        def __init__(self):
            super(MyNet, self).__init__()
            self.emb = paddle.nn.Embedding(vocab_size, emb_size)
            self.fc = paddle.nn.Linear(in_features=emb_size, out_features=2)
            self.dropout = paddle.nn.Dropout(0.5)
    
        def forward(self, x):
            x = self.emb(x)
            x = paddle.reduce_mean(x, dim=1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

开始模型的训练
--------------

.. code:: ipython3

    def train(model):
        model.train()
    
        opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    
        for epoch in range(epoch_num):
            # shuffle data
            perm = np.random.permutation(len(train_sents))
            train_sents_shuffled = train_sents[perm]
            train_labels_shuffled = train_labels[perm]
            
            for batch_id in range(len(train_sents_shuffled) // batch_size):
                x_data = train_sents_shuffled[(batch_id * batch_size):((batch_id+1)*batch_size)]
                y_data = train_labels_shuffled[(batch_id * batch_size):((batch_id+1)*batch_size)]
                
                sent = paddle.to_tensor(x_data)
                label = paddle.to_tensor(y_data)
                
                logits = model(sent)
                loss = paddle.nn.functional.softmax_with_cross_entropy(logits, label)
                
                avg_loss = paddle.mean(loss)
                if batch_id % 500 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
                avg_loss.backward()
                opt.step()
                opt.clear_grad()
    
            # evaluate model after one epoch
            model.eval()
            accuracies = []
            losses = []
            for batch_id in range(len(test_sents) // batch_size):
                x_data = test_sents[(batch_id * batch_size):((batch_id+1)*batch_size)]
                y_data = test_labels[(batch_id * batch_size):((batch_id+1)*batch_size)]
            
                sent = paddle.to_tensor(x_data)
                label = paddle.to_tensor(y_data)
    
                logits = model(sent)
                loss = paddle.nn.functional.softmax_with_cross_entropy(logits, label)
                acc = paddle.metric.accuracy(logits, label)
                
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())
            
            avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
            print("[validation] accuracy/loss: {}/{}".format(avg_acc, avg_loss))
            
            model.train()
            
    model = MyNet()
    train(model)


.. parsed-literal::

    epoch: 0, batch_id: 0, loss is: [0.6918494]
    epoch: 0, batch_id: 500, loss is: [0.33142853]
    [validation] accuracy/loss: 0.8506321907043457/0.3620821535587311
    epoch: 1, batch_id: 0, loss is: [0.37161]
    epoch: 1, batch_id: 500, loss is: [0.2296829]
    [validation] accuracy/loss: 0.8622759580612183/0.3286365270614624


The End
-------

可以看到，在这个数据集上，经过两轮的迭代可以得到86%左右的准确率。你也可以通过调整网络结构和超参数，来获得更好的效果。
