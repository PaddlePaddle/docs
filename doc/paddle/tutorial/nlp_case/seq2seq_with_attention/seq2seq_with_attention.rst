使用注意力机制的LSTM的机器翻译
==============================

本示例教程介绍如何使用飞桨完成一个机器翻译任务。我们将会使用飞桨提供的LSTM的API，组建一个\ ``sequence to sequence with attention``\ 的机器翻译的模型，并在示例的数据集上完成从英文翻译成中文的机器翻译。

环境设置
--------

本示例教程基于飞桨2.0RC版本。

.. code:: ipython3

    import paddle
    import paddle.nn.functional as F
    import re
    import numpy as np
    
    print(paddle.__version__)


.. parsed-literal::

    2.0.0-rc0


下载数据集
----------

我们将使用 http://www.manythings.org/anki/
提供的中英文的英汉句对作为数据集，来完成本任务。该数据集含有23610个中英文双语的句对。

.. code:: ipython3

    !wget -c https://www.manythings.org/anki/cmn-eng.zip && unzip cmn-eng.zip


.. parsed-literal::

    --2020-10-26 09:50:14--  https://www.manythings.org/anki/cmn-eng.zip
    Resolving www.manythings.org (www.manythings.org)... 2606:4700:3033::6818:6dc4, 2606:4700:3036::ac43:adc6, 2606:4700:3037::6818:6cc4, ...
    Connecting to www.manythings.org (www.manythings.org)|2606:4700:3033::6818:6dc4|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1030722 (1007K) [application/zip]
    Saving to: ‘cmn-eng.zip’
    
    cmn-eng.zip         100%[===================>]   1007K   138KB/s    in 7.3s    
    
    2020-10-26 09:50:23 (138 KB/s) - ‘cmn-eng.zip’ saved [1030722/1030722]
    
    Archive:  cmn-eng.zip
      inflating: cmn.txt                 
      inflating: _about.txt              


.. code:: ipython3

    !wc -l cmn.txt


.. parsed-literal::

       23610 cmn.txt


构建双语句对的数据结构
----------------------

接下来我们通过处理下载下来的双语句对的文本文件，将双语句对读入到python的数据结构中。这里做了如下的处理。

-  对于英文，会把全部英文都变成小写，并只保留英文的单词。
-  对于中文，为了简便起见，未做分词，按照字做了切分。
-  为了后续的程序运行的更快，我们通过限制句子长度，和只保留部分英文单词开头的句子的方式，得到了一个较小的数据集。这样得到了一个有5508个句对的数据集。

.. code:: ipython3

    MAX_LEN = 10

.. code:: ipython3

    lines = open('cmn.txt', encoding='utf-8').read().strip().split('\n')
    words_re = re.compile(r'\w+')
    
    pairs = []
    for l in lines:
        en_sent, cn_sent, _ = l.split('\t')
        pairs.append((words_re.findall(en_sent.lower()), list(cn_sent)))
    
    # create a smaller dataset to make the demo process faster
    filtered_pairs = []
    
    for x in pairs:
        if len(x[0]) < MAX_LEN and len(x[1]) < MAX_LEN and \
        x[0][0] in ('i', 'you', 'he', 'she', 'we', 'they'):
            filtered_pairs.append(x)
               
    print(len(filtered_pairs))
    for x in filtered_pairs[:10]: print(x) 


.. parsed-literal::

    5508
    (['i', 'won'], ['我', '赢', '了', '。'])
    (['he', 'ran'], ['他', '跑', '了', '。'])
    (['i', 'quit'], ['我', '退', '出', '。'])
    (['i', 'm', 'ok'], ['我', '沒', '事', '。'])
    (['i', 'm', 'up'], ['我', '已', '经', '起', '来', '了', '。'])
    (['we', 'try'], ['我', '们', '来', '试', '试', '。'])
    (['he', 'came'], ['他', '来', '了', '。'])
    (['he', 'runs'], ['他', '跑', '。'])
    (['i', 'agree'], ['我', '同', '意', '。'])
    (['i', 'm', 'ill'], ['我', '生', '病', '了', '。'])


创建词表
--------

接下来我们分别创建中英文的词表，这两份词表会用来将英文和中文的句子转换为词的ID构成的序列。词表中还加入了如下三个特殊的词：
- ``<pad>``: 用来对较短的句子进行填充。 - ``<bos>``: “begin of
sentence”， 表示句子的开始的特殊词。 - ``<eos>``: “end of sentence”，
表示句子的结束的特殊词。

Note:
在实际的任务中，可能还需要通过\ ``<unk>``\ （或者\ ``<oov>``\ ）特殊词来表示未在词表中出现的词。

.. code:: ipython3

    en_vocab = {}
    cn_vocab = {}
    
    # create special token for pad, begin of sentence, end of sentence
    en_vocab['<pad>'], en_vocab['<bos>'], en_vocab['<eos>'] = 0, 1, 2
    cn_vocab['<pad>'], cn_vocab['<bos>'], cn_vocab['<eos>'] = 0, 1, 2
    
    en_idx, cn_idx = 3, 3
    for en, cn in filtered_pairs:
        for w in en: 
            if w not in en_vocab: 
                en_vocab[w] = en_idx
                en_idx += 1
        for w in cn:  
            if w not in cn_vocab: 
                cn_vocab[w] = cn_idx
                cn_idx += 1
    
    print(len(list(en_vocab)))
    print(len(list(cn_vocab)))


.. parsed-literal::

    2539
    2039


创建padding过的数据集
---------------------

接下来根据词表，我们将会创建一份实际的用于训练的用numpy
array组织起来的数据集。 -
所有的句子都通过\ ``<pad>``\ 补充成为了长度相同的句子。 -
对于英文句子（源语言），我们将其反转了过来，这会带来更好的翻译的效果。 -
所创建的\ ``padded_cn_label_sents``\ 是训练过程中的预测的目标，即，每个中文的当前词去预测下一个词是什么词。

.. code:: ipython3

    padded_en_sents = []
    padded_cn_sents = []
    padded_cn_label_sents = []
    for en, cn in filtered_pairs:
        # reverse source sentence
        padded_en_sent = en + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(en))
        padded_en_sent.reverse()
        padded_cn_sent = ['<bos>'] + cn + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(cn))
        padded_cn_label_sent = cn + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(cn) + 1) 
    
        padded_en_sents.append([en_vocab[w] for w in padded_en_sent])
        padded_cn_sents.append([cn_vocab[w] for w in padded_cn_sent])
        padded_cn_label_sents.append([cn_vocab[w] for w in padded_cn_label_sent])
    
    train_en_sents = np.array(padded_en_sents)
    train_cn_sents = np.array(padded_cn_sents)
    train_cn_label_sents = np.array(padded_cn_label_sents)
    
    print(train_en_sents.shape)
    print(train_cn_sents.shape)
    print(train_cn_label_sents.shape)


.. parsed-literal::

    (5508, 11)
    (5508, 12)
    (5508, 12)


创建网络
--------

我们将会创建一个Encoder-AttentionDecoder架构的模型结构用来完成机器翻译任务。
首先我们将设置一些必要的网络结构中用到的参数。

.. code:: ipython3

    embedding_size = 128
    hidden_size = 256
    num_encoder_lstm_layers = 1
    en_vocab_size = len(list(en_vocab))
    cn_vocab_size = len(list(cn_vocab))
    epochs = 20
    batch_size = 16

Encoder部分
-----------

在编码器的部分，我们通过查找完Embedding之后接一个LSTM的方式构建一个对源语言编码的网络。飞桨的RNN系列的API，除了LSTM之外，还提供了SimleRNN,
GRU供使用，同时，还可以使用反向RNN，双向RNN，多层RNN等形式。也可以通过\ ``dropout``\ 参数设置是否对多层RNN的中间层进行\ ``dropout``\ 处理，来防止过拟合。

除了使用序列到序列的RNN操作之外，也可以通过SimpleRNN, GRUCell,
LSTMCell等API更灵活的创建单步的RNN计算，甚至通过继承RNNCellBase来实现自己的RNN计算单元。

.. code:: ipython3

    # encoder: simply learn representation of source sentence
    class Encoder(paddle.nn.Layer):
        def __init__(self):
            super(Encoder, self).__init__()
            self.emb = paddle.nn.Embedding(en_vocab_size, embedding_size,)
            self.lstm = paddle.nn.LSTM(input_size=embedding_size, 
                                       hidden_size=hidden_size, 
                                       num_layers=num_encoder_lstm_layers)
    
        def forward(self, x):
            x = self.emb(x)
            x, (_, _) = self.lstm(x)
            return x

AttentionDecoder部分
--------------------

在解码器部分，我们通过一个带有注意力机制的LSTM来完成解码。

-  单步的LSTM：在解码器的实现的部分，我们同样使用LSTM，与Encoder部分不同的是，下面的代码，每次只让LSTM往前计算一次。整体的recurrent部分，是在训练循环内完成的。
-  注意力机制：这里使用了一个由两个Linear组成的网络来完成注意力机制的计算，它用来计算出目标语言在每次翻译一个词的时候，需要对源语言当中的每个词需要赋予多少的权重。
-  对于第一次接触这样的网络结构来说，下面的代码在理解起来可能稍微有些复杂，你可以通过插入打印每个tensor在不同步骤时的形状的方式来更好的理解。

.. code:: ipython3

    # only move one step of LSTM, 
    # the recurrent loop is implemented inside training loop
    class AttentionDecoder(paddle.nn.Layer):
        def __init__(self):
            super(AttentionDecoder, self).__init__()
            self.emb = paddle.nn.Embedding(cn_vocab_size, embedding_size)
            self.lstm = paddle.nn.LSTM(input_size=embedding_size + hidden_size, 
                                       hidden_size=hidden_size)
    
            # for computing attention weights
            self.attention_linear1 = paddle.nn.Linear(hidden_size * 2, hidden_size)
            self.attention_linear2 = paddle.nn.Linear(hidden_size, 1)
            
            # for computing output logits
            self.outlinear =paddle.nn.Linear(hidden_size, cn_vocab_size)
    
        def forward(self, x, previous_hidden, previous_cell, encoder_outputs):
            x = self.emb(x)
            
            attention_inputs = paddle.concat((encoder_outputs, 
                                          paddle.tile(previous_hidden, repeat_times=[1, MAX_LEN+1, 1])),
                                          axis=-1
                                         )
    
            attention_hidden = self.attention_linear1(attention_inputs)
            attention_hidden = F.tanh(attention_hidden)
            attention_logits = self.attention_linear2(attention_hidden)
            attention_logits = paddle.squeeze(attention_logits)
    
            attention_weights = F.softmax(attention_logits)        
            attention_weights = paddle.expand_as(paddle.unsqueeze(attention_weights, -1), 
                                                 encoder_outputs)
    
            context_vector = paddle.multiply(encoder_outputs, attention_weights)               
            context_vector = paddle.sum(context_vector, 1)
            context_vector = paddle.unsqueeze(context_vector, 1)
            
            lstm_input = paddle.concat((x, context_vector), axis=-1)
    
            # LSTM requirement to previous hidden/state: 
            # (number_of_layers * direction, batch, hidden)
            previous_hidden = paddle.transpose(previous_hidden, [1, 0, 2])
            previous_cell = paddle.transpose(previous_cell, [1, 0, 2])
            
            x, (hidden, cell) = self.lstm(lstm_input, (previous_hidden, previous_cell))
            
            # change the return to (batch, number_of_layers * direction, hidden)
            hidden = paddle.transpose(hidden, [1, 0, 2])
            cell = paddle.transpose(cell, [1, 0, 2])
    
            output = self.outlinear(hidden)
            output = paddle.squeeze(output)
            return output, (hidden, cell)

训练模型
--------

接下来我们开始训练模型。

-  在每个epoch开始之前，我们对训练数据进行了随机打乱。
-  我们通过多次调用\ ``atten_decoder``\ ，在这里实现了解码时的recurrent循环。
-  ``teacher forcing``\ 策略:
   在每次解码下一个词时，我们给定了训练数据当中的真实词作为了预测下一个词时的输入。相应的，你也可以尝试用模型预测的结果作为下一个词的输入。（或者混合使用）

.. code:: ipython3

    encoder = Encoder()
    atten_decoder = AttentionDecoder()
    
    opt = paddle.optimizer.Adam(learning_rate=0.001, 
                                parameters=encoder.parameters()+atten_decoder.parameters())
    
    for epoch in range(epochs):
        print("epoch:{}".format(epoch))
    
        # shuffle training data
        perm = np.random.permutation(len(train_en_sents))
        train_en_sents_shuffled = train_en_sents[perm]
        train_cn_sents_shuffled = train_cn_sents[perm]
        train_cn_label_sents_shuffled = train_cn_label_sents[perm]
    
        for iteration in range(train_en_sents_shuffled.shape[0] // batch_size):
            x_data = train_en_sents_shuffled[(batch_size*iteration):(batch_size*(iteration+1))]
            sent = paddle.to_tensor(x_data)
            en_repr = encoder(sent)
    
            x_cn_data = train_cn_sents_shuffled[(batch_size*iteration):(batch_size*(iteration+1))]
            x_cn_label_data = train_cn_label_sents_shuffled[(batch_size*iteration):(batch_size*(iteration+1))]
    
            # shape: (batch,  num_layer(=1 here) * num_of_direction(=1 here), hidden_size)
            hidden = paddle.zeros([batch_size, 1, hidden_size])
            cell = paddle.zeros([batch_size, 1, hidden_size])
    
            loss = paddle.zeros([1])
            # the decoder recurrent loop mentioned above
            for i in range(MAX_LEN + 2):
                cn_word = paddle.to_tensor(x_cn_data[:,i:i+1])
                cn_word_label = paddle.to_tensor(x_cn_label_data[:,i])
    
                logits, (hidden, cell) = atten_decoder(cn_word, hidden, cell, en_repr)
                step_loss = F.cross_entropy(logits, cn_word_label)
                loss += step_loss
    
            loss = loss / (MAX_LEN + 2)
            if(iteration % 200 == 0):
                print("iter {}, loss:{}".format(iteration, loss.numpy()))
    
            loss.backward()
            opt.step()
            opt.clear_grad()


.. parsed-literal::

    epoch:0
    iter 0, loss:[7.6254287]
    iter 200, loss:[2.7549095]
    epoch:1
    iter 0, loss:[3.25681]
    iter 200, loss:[3.1060884]
    epoch:2
    iter 0, loss:[2.8566368]
    iter 200, loss:[2.5701585]
    epoch:3
    iter 0, loss:[2.5982018]
    iter 200, loss:[2.498022]
    epoch:4
    iter 0, loss:[2.4150505]
    iter 200, loss:[2.2246962]
    epoch:5
    iter 0, loss:[2.2809484]
    iter 200, loss:[2.0454435]
    epoch:6
    iter 0, loss:[1.9620974]
    iter 200, loss:[1.9354618]
    epoch:7
    iter 0, loss:[1.404521]
    iter 200, loss:[1.6144934]
    epoch:8
    iter 0, loss:[1.6302392]
    iter 200, loss:[1.6218137]
    epoch:9
    iter 0, loss:[1.6828392]
    iter 200, loss:[1.7782025]
    epoch:10
    iter 0, loss:[1.1777062]
    iter 200, loss:[1.2404836]
    epoch:11
    iter 0, loss:[1.2056196]
    iter 200, loss:[1.322629]
    epoch:12
    iter 0, loss:[1.316817]
    iter 200, loss:[1.021146]
    epoch:13
    iter 0, loss:[1.2051158]
    iter 200, loss:[1.227415]
    epoch:14
    iter 0, loss:[1.0421599]
    iter 200, loss:[0.7064129]
    epoch:15
    iter 0, loss:[0.7054539]
    iter 200, loss:[1.1203959]
    epoch:16
    iter 0, loss:[0.7972643]
    iter 200, loss:[0.57451296]
    epoch:17
    iter 0, loss:[0.5825621]
    iter 200, loss:[0.66827786]
    epoch:18
    iter 0, loss:[0.5396042]
    iter 200, loss:[0.60596395]
    epoch:19
    iter 0, loss:[0.41747904]
    iter 200, loss:[0.58902776]


使用模型进行机器翻译
--------------------

根据你所使用的计算设备的不同，上面的训练过程可能需要不等的时间。（在一台Mac笔记本上，大约耗时15~20分钟）
完成上面的模型训练之后，我们可以得到一个能够从英文翻译成中文的机器翻译模型。接下来我们通过一个greedy
search来实现使用该模型完成实际的机器翻译。（实际的任务中，你可能需要用beam
search算法来提升效果）

.. code:: ipython3

    encoder.eval()
    atten_decoder.eval()
    
    num_of_exampels_to_evaluate = 10
    
    indices = np.random.choice(len(train_en_sents),  num_of_exampels_to_evaluate, replace=False)
    x_data = train_en_sents[indices]
    sent = paddle.to_tensor(x_data)
    en_repr = encoder(sent)
    
    word = np.array(
        [[cn_vocab['<bos>']]] * num_of_exampels_to_evaluate
    )
    word = paddle.to_tensor(word)
    
    hidden = paddle.zeros([num_of_exampels_to_evaluate, 1, hidden_size])
    cell = paddle.zeros([num_of_exampels_to_evaluate, 1, hidden_size])
    
    decoded_sent = []
    for i in range(MAX_LEN + 2):
        logits, (hidden, cell) = atten_decoder(word, hidden, cell, en_repr)
        word = paddle.argmax(logits, axis=1)
        decoded_sent.append(word.numpy())
        word = paddle.unsqueeze(word, axis=-1)
        
    results = np.stack(decoded_sent, axis=1)
    for i in range(num_of_exampels_to_evaluate):
        en_input = " ".join(filtered_pairs[indices[i]][0])
        ground_truth_translate = "".join(filtered_pairs[indices[i]][1])
        model_translate = ""
        for k in results[i]:
            w = list(cn_vocab)[k]
            if w != '<pad>' and w != '<eos>':
                model_translate += w
        print(en_input)
        print("true: {}".format(ground_truth_translate))
        print("pred: {}".format(model_translate))


.. parsed-literal::

    he may have been ill
    true: 他可能病了。
    pred: 他可能生病了。
    she kept working
    true: 她继续工作。
    pred: 她继续工作。
    i don t think he ll come
    true: 我不認為他會來的。
    pred: 我不認為他會來的。
    we all have secrets
    true: 我們都有秘密。
    pred: 我們都有秘密。
    he s the same age as me
    true: 他和我同岁。
    pred: 他和我同岁。
    he can speak russian as well
    true: 他还会说俄语。
    pred: 他也會說俄語。
    he is a thief
    true: 这是一个小偷。
    pred: 他是義一個人人。
    i forgot to ask him
    true: 我忘了問他。
    pred: 我忘了他。
    i believe you
    true: 我相信你。
    pred: 我相信你。
    i feel a whole lot better today
    true: 我今天感觉好多了。
    pred: 我今天感觉好多了。


The End
-------

你还可以通过变换网络结构，调整数据集，尝试不同的参数的方式来进一步提升本示例当中的机器翻译的效果。同时，也可以尝试在其他的类似的任务中用飞桨来完成实际的实践。
