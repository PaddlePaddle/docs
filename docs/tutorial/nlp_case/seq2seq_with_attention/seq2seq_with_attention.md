# 使用注意力机制的LSTM的机器翻译

**作者:** [PaddlePaddle](https://github.com/PaddlePaddle) <br>
**日期:** 2021.05 <br>
**摘要:** 本示例教程介绍如何使用飞桨完成一个机器翻译任务。通过使用飞桨提供的LSTM的API，组建一个`sequence to sequence with attention`的机器翻译的模型，并在示例的数据集上完成从英文翻译成中文的机器翻译。

## 一、环境配置

本教程基于Paddle 2.1 编写，如果你的环境不是本版本，请先参考官网[安装](https://www.paddlepaddle.org.cn/install/quick) Paddle 2.1 。


```python
import paddle
import paddle.nn.functional as F
import re
import numpy as np

print(paddle.__version__)
```

    2.1.0


## 二、数据加载

### 2.1 数据集下载

将使用 [http://www.manythings.org/anki/](http://www.manythings.org/anki/) 提供的中英文的英汉句对作为数据集，来完成本任务。该数据集含有23610个中英文双语的句对。


```python
!wget -c https://www.manythings.org/anki/cmn-eng.zip && unzip cmn-eng.zip
```

    --2021-05-18 18:00:27--  https://www.manythings.org/anki/cmn-eng.zip
    Resolving www.manythings.org (www.manythings.org)... 172.67.173.198, 104.21.55.222, 2606:4700:3031::6815:37de, ...
    Connecting to www.manythings.org (www.manythings.org)|172.67.173.198|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1062383 (1.0M) [application/zip]
    Saving to: ‘cmn-eng.zip’
    
    cmn-eng.zip         100%[===================>]   1.01M   902KB/s    in 1.1s    
    
    2021-05-18 18:00:29 (902 KB/s) - ‘cmn-eng.zip’ saved [1062383/1062383]
    
    Archive:  cmn-eng.zip
      inflating: cmn.txt                 
      inflating: _about.txt              



```python
!wc -l cmn.txt
```

    24360 cmn.txt


### 2.2 构建双语句对的数据结构

接下来通过处理下载下来的双语句对的文本文件，将双语句对读入到python的数据结构中。这里做了如下的处理。

- 对于英文，会把全部英文都变成小写，并只保留英文的单词。
- 对于中文，为了简便起见，未做分词，按照字做了切分。
- 为了后续的程序运行的更快，通过限制句子长度，和只保留部分英文单词开头的句子的方式，得到了一个较小的数据集。这样得到了一个有5508个句对的数据集。


```python
MAX_LEN = 10
```


```python
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
```

    5687
    (['i', 'won'], ['我', '赢', '了', '。'])
    (['he', 'ran'], ['他', '跑', '了', '。'])
    (['i', 'quit'], ['我', '退', '出', '。'])
    (['i', 'quit'], ['我', '不', '干', '了', '。'])
    (['i', 'm', 'ok'], ['我', '沒', '事', '。'])
    (['i', 'm', 'up'], ['我', '已', '经', '起', '来', '了', '。'])
    (['we', 'try'], ['我', '们', '来', '试', '试', '。'])
    (['he', 'came'], ['他', '来', '了', '。'])
    (['he', 'runs'], ['他', '跑', '。'])
    (['i', 'agree'], ['我', '同', '意', '。'])


### 2.3 创建词表

接下来分别创建中英文的词表，这两份词表会用来将英文和中文的句子转换为词的ID构成的序列。词表中还加入了如下三个特殊的词：
- `<pad>`: 用来对较短的句子进行填充。
- `<bos>`: "begin of sentence"， 表示句子的开始的特殊词。
- `<eos>`: "end of sentence"， 表示句子的结束的特殊词。

Note: 在实际的任务中，可能还需要通过`<unk>`（或者`<oov>`）特殊词来表示未在词表中出现的词。


```python
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
```

    2584
    2055


### 2.4 创建padding过的数据集

接下来根据词表，将会创建一份实际的用于训练的用numpy array组织起来的数据集。
- 所有的句子都通过`<pad>`补充成为了长度相同的句子。
- 对于英文句子（源语言），将其反转了过来，这会带来更好的翻译的效果。
- 所创建的`padded_cn_label_sents`是训练过程中的预测的目标，即，每个中文的当前词去预测下一个词是什么词。



```python
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
```

    (5687, 11)
    (5687, 12)
    (5687, 12)


## 三、网络构建

将会创建一个Encoder-AttentionDecoder架构的模型结构用来完成机器翻译任务。

首先将设置一些必要的网络结构中用到的参数。


```python
embedding_size = 128
hidden_size = 256
num_encoder_lstm_layers = 1
en_vocab_size = len(list(en_vocab))
cn_vocab_size = len(list(cn_vocab))
epochs = 20
batch_size = 16
```

### 3.1 Encoder部分

在编码器的部分，通过查找完Embedding之后接一个LSTM的方式构建一个对源语言编码的网络。飞桨的RNN系列的API，除了LSTM之外，还提供了SimleRNN, GRU供使用，同时，还可以使用反向RNN，双向RNN，多层RNN等形式。也可以通过`dropout`参数设置是否对多层RNN的中间层进行`dropout`处理，来防止过拟合。

除了使用序列到序列的RNN操作之外，也可以通过SimpleRNN, GRUCell, LSTMCell等API更灵活的创建单步的RNN计算，甚至通过继承RNNCellBase来实现自己的RNN计算单元。


```python
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
```

### 3.2 AttentionDecoder部分

在解码器部分，通过一个带有注意力机制的LSTM来完成解码。

- 单步的LSTM：在解码器的实现的部分，同样使用LSTM，与Encoder部分不同的是，下面的代码，每次只让LSTM往前计算一次。整体的recurrent部分，是在训练循环内完成的。
- 注意力机制：这里使用了一个由两个Linear组成的网络来完成注意力机制的计算，它用来计算出目标语言在每次翻译一个词的时候，需要对源语言当中的每个词需要赋予多少的权重。
- 对于第一次接触这样的网络结构来说，下面的代码在理解起来可能稍微有些复杂，你可以通过插入打印每个tensor在不同步骤时的形状的方式来更好的理解。


```python
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
```

## 四、训练模型

接下来开始训练模型。

- 在每个epoch开始之前，对训练数据进行了随机打乱。
- 通过多次调用`atten_decoder`，在这里实现了解码时的recurrent循环。
- `teacher forcing`策略: 在每次解码下一个词时，给定了训练数据当中的真实词作为了预测下一个词时的输入。相应的，你也可以尝试用模型预测的结果作为下一个词的输入。（或者混合使用）


```python
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
```

    epoch:0
    iter 0, loss:[7.6239414]
    iter 200, loss:[3.0424228]
    epoch:1
    iter 0, loss:[3.3344]
    iter 200, loss:[3.135097]
    epoch:2
    iter 0, loss:[2.8372972]
    iter 200, loss:[2.9874132]
    epoch:3
    iter 0, loss:[2.4977226]
    iter 200, loss:[2.347312]
    epoch:4
    iter 0, loss:[2.1353514]
    iter 200, loss:[2.1293092]
    epoch:5
    iter 0, loss:[2.0924835]
    iter 200, loss:[2.0193372]
    epoch:6
    iter 0, loss:[1.9638586]
    iter 200, loss:[1.9775124]
    epoch:7
    iter 0, loss:[1.8319316]
    iter 200, loss:[1.6078386]
    epoch:8
    iter 0, loss:[1.5061388]
    iter 200, loss:[1.4000171]
    epoch:9
    iter 0, loss:[1.6096058]
    iter 200, loss:[1.5511936]
    epoch:10
    iter 0, loss:[1.4739537]
    iter 200, loss:[1.2266061]
    epoch:11
    iter 0, loss:[1.3779855]
    iter 200, loss:[1.3114413]
    epoch:12
    iter 0, loss:[1.146878]
    iter 200, loss:[1.2473543]
    epoch:13
    iter 0, loss:[1.0685896]
    iter 200, loss:[1.1657724]
    epoch:14
    iter 0, loss:[0.87895143]
    iter 200, loss:[0.8481859]
    epoch:15
    iter 0, loss:[0.77661693]
    iter 200, loss:[0.76900244]
    epoch:16
    iter 0, loss:[0.8463232]
    iter 200, loss:[0.8217341]
    epoch:17
    iter 0, loss:[0.5898639]
    iter 200, loss:[0.6576902]
    epoch:18
    iter 0, loss:[0.51171196]
    iter 200, loss:[0.53207105]
    epoch:19
    iter 0, loss:[0.49052936]
    iter 200, loss:[0.5290806]


## 五、使用模型进行机器翻译

根据你所使用的计算设备的不同，上面的训练过程可能需要不等的时间。（在一台Mac笔记本上，大约耗时15~20分钟）

完成上面的模型训练之后，可以得到一个能够从英文翻译成中文的机器翻译模型。接下来通过一个greedy search来实现使用该模型完成实际的机器翻译。（实际的任务中，你可能需要用beam search算法来提升效果）


```python
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
```

    she died of stomach cancer
    true: 她死于胃癌。
    pred: 她死于胃癌。
    we know
    true: 我们知道。
    pred: 我们知道。
    he applied for the scholarship
    true: 他申請了獎學金。
    pred: 他申請了獎學門。
    he went bankrupt
    true: 他破产了。
    pred: 他破产了。
    i ll try harder next time
    true: 下次我會更加努力。
    pred: 我會去学习几次。
    he tires easily
    true: 他很容易觉得累。
    pred: 他很容易觉得累。
    i ll do what i can to help you
    true: 我會盡力幫你。
    pred: 我會盡力幫你。
    you ought to have come here earlier
    true: 你應該早點來的。
    pred: 你最好去睡觉。
    i m going to change my shirt
    true: 我要去換我的襯衫。
    pred: 我要去換我的襯衫。
    i almost didn t meet her
    true: 我幾乎沒有遇見她。
    pred: 我幾乎沒有信她。


## The End

你还可以通过变换网络结构，调整数据集，尝试不同的参数的方式来进一步提升本示例当中的机器翻译的效果。同时，也可以尝试在其他的类似的任务中用飞桨来完成实际的实践。
