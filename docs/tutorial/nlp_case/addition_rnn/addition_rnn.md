# 使用序列到序列模型完成数字加法

**作者:** [jm12138](https://github.com/jm12138) <br>
**日期:** 2021.10 <br>
**摘要:** 本示例介绍如何使用飞桨完成一个数字加法任务，将会使用飞桨提供的`LSTM`，组建一个序列到序列模型，并在随机生成的数据集上完成数字加法任务的模型训练与预测。

## 一、环境配置

本教程基于Paddle 2.2.0-rc0 编写，如果你的环境不是本版本，请先参考官网[安装](https://www.paddlepaddle.org.cn/install/quick) Paddle 2.2.0-rc0 。


```python
# 导入项目运行所需的包

import paddle
import paddle.nn as nn

import random
import numpy as np

from visualdl import LogWriter

# 打印Paddle版本
print('paddle version: %s' % paddle.__version__)
```

    paddle version: 2.2.0-rc0


## 二、构建数据集
* 随机生成数据，并使用生成的数据构造数据集
* 通过继承 ``paddle.io.Dataset`` 来完成数据集的构造


```python
# 编码函数
def encoder(text, LEN, label_dict):
    # 文本转ID
    ids = [label_dict[word] for word in text]
    # 对长度进行补齐
    ids += [label_dict[' ']]*(LEN-len(ids))
    return ids

# 单个数据生成函数
def make_data(inputs, labels, DIGITS, label_dict):
    MAXLEN = DIGITS + 1 + DIGITS
    # 对输入输出文本进行ID编码
    inputs = encoder(inputs, MAXLEN, label_dict)
    labels = encoder(labels, DIGITS + 1, label_dict)
    return inputs, labels

# 批量数据生成函数
def gen_datas(DATA_NUM, MAX_NUM, DIGITS, label_dict):
    datas = []
    while len(datas)<DATA_NUM:
        # 随机取两个数
        a = random.randint(0,MAX_NUM)
        b = random.randint(0,MAX_NUM)
        # 生成输入文本
        inputs = '%d+%d' % (a, b)
        # 生成输出文本
        labels = str(eval(inputs))
        # 生成单个数据
        inputs, labels = [np.array(_).astype('int64') for _ in make_data(inputs, labels, DIGITS, label_dict)]
        datas.append([inputs, labels])
    return datas

# 继承paddle.io.Dataset来构造数据集
class Addition_Dataset(paddle.io.Dataset):
    # 重写数据集初始化函数
    def __init__(self, datas):
        super(Addition_Dataset, self).__init__()
        self.datas = datas
    
    # 重写生成样本的函数
    def __getitem__(self, index):
        data, label = [paddle.to_tensor(_) for _ in self.datas[index]]
        return data, label

    # 重写返回数据集大小的函数
    def __len__(self):
        return len(self.datas)

print('generating datas..')

# 定义字符表
label_dict = {
    '0': 0, '1': 1, '2': 2, '3': 3,
    '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '+': 10, ' ': 11
}

# 输入数字最大位数
DIGITS = 2

# 数据数量
train_num = 5000
dev_num = 500

# 数据批大小
batch_size = 32

# 读取线程数
num_workers = 8

# 定义一些所需变量
MAXLEN = DIGITS + 1 + DIGITS
MAX_NUM = 10**(DIGITS)-1

# 生成数据
train_datas = gen_datas(
    train_num, 
    MAX_NUM,
    DIGITS, 
    label_dict
) 
dev_datas = gen_datas(
    dev_num, 
    MAX_NUM,
    DIGITS, 
    label_dict
)

# 实例化数据集
train_dataset = Addition_Dataset(train_datas)
dev_dataset = Addition_Dataset(dev_datas)

print('making the dataset...')

# 实例化数据读取器
train_reader = paddle.io.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)
dev_reader = paddle.io.DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
)

print('finish')
```

    generating datas..
    making the dataset...
    finish


##  三、模型组网
* 通过继承 ``paddle.nn.Layer`` 类来搭建模型
* 本次介绍的模型是一个简单的基于 ``LSTM`` 的 ``Seq2Seq`` 模型
* 一共有如下四个主要的网络层：

  1. 嵌入层(``Embedding``)：将输入的文本序列转为嵌入向量
  2. 编码层(``LSTM``)：将嵌入向量进行编码
  3. 解码层(``LSTM``)：将编码向量进行解码
  4. 全连接层(``Linear``)：对解码完成的向量进行线性映射
* 损失函数为交叉熵损失函数


```python
# 继承paddle.nn.Layer类
class Addition_Model(nn.Layer):
    # 重写初始化函数
    # 参数：字符表长度、嵌入层大小、隐藏层大小、解码器层数、处理数字的最大位数
    def __init__(self, char_len=12, embedding_size=128, hidden_size=128, num_layers=1, DIGITS=2):
        super(Addition_Model, self).__init__()
        # 初始化变量
        self.DIGITS = DIGITS
        self.MAXLEN = DIGITS + 1 + DIGITS
        self.hidden_size = hidden_size
        self.char_len = char_len

        # 嵌入层
        self.emb = nn.Embedding(
            char_len, 
            embedding_size
        )
        
        # 编码器
        self.encoder = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # 全连接层
        self.fc = nn.Linear(
            hidden_size, 
            char_len
        )
    
    # 重写模型前向计算函数
    # 参数：输入[None, MAXLEN]、标签[None, DIGITS + 1]
    def forward(self, inputs, labels=None):
        # 嵌入层
        out = self.emb(inputs)

        # 编码器
        out, (_, _) = self.encoder(out)

        # 按时间步切分编码器输出
        out = paddle.split(out, self.MAXLEN, axis=1)

        # 取最后一个时间步的输出并复制 DIGITS + 1 次
        out = paddle.expand(out[-1], [out[-1].shape[0], self.DIGITS + 1, self.hidden_size])

        # 解码器
        out, (_, _) = self.decoder(out)

        # 全连接
        out = self.fc(out)

        # 如果标签存在，则计算其损失和准确率
        if labels is not None:

            # 计算交叉熵损失
            loss = nn.functional.cross_entropy(out, labels)

            # 计算准确率
            acc = paddle.metric.accuracy(paddle.reshape(out, [-1, self.char_len]), paddle.reshape(labels, [-1, 1]))

            # 返回损失和准确率
            return loss, acc

        # 返回输出
        return out
```

## 四、模型训练与评估
* 使用 ``Adam`` 作为优化器进行模型训练
* 以模型准确率作为评价指标
* 使用 ``VisualDL`` 对训练数据进行可视化
* 训练过程中会同时进行模型评估和最佳模型的保存


```python
# 初始化log写入器
log_writer = LogWriter(logdir="./log")

# 模型参数设置
embedding_size = 128
hidden_size=128
num_layers=1

# 训练参数设置
epoch_num = 50
learning_rate = 0.001
log_iter = 2000
eval_iter = 500

# 定义一些所需变量
global_step = 0
log_step = 0
max_acc = 0

# 实例化模型
model = Addition_Model(
    char_len=len(label_dict), 
    embedding_size=embedding_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers, 
    DIGITS=DIGITS)

# 将模型设置为训练模式
model.train()

# 设置优化器，学习率，并且把模型参数给优化器
opt = paddle.optimizer.Adam(
    learning_rate=learning_rate,
    parameters=model.parameters()
)

# 启动训练，循环epoch_num个轮次
for epoch in range(epoch_num):
    # 遍历数据集读取数据
    for batch_id, data in enumerate(train_reader()):
        # 读取数据
        inputs, labels = data

        # 模型前向计算
        loss, acc = model(inputs, labels=labels)

        # 打印训练数据
        if global_step%log_iter==0:
            print('train epoch:%d step: %d loss:%f acc:%f' % (epoch, global_step, loss.numpy(), acc.numpy()))
            log_writer.add_scalar(tag="train/loss", step=log_step, value=loss.numpy())
            log_writer.add_scalar(tag="train/acc", step=log_step, value=acc.numpy())
            log_step+=1

        # 模型验证
        if global_step%eval_iter==0:
            model.eval()
            losses = []
            accs = []
            for data in dev_reader():
                loss_eval, acc_eval = model(inputs, labels=labels)
                losses.append(loss_eval.numpy())
                accs.append(acc_eval.numpy())
            avg_loss = np.concatenate(losses).mean()
            avg_acc = np.concatenate(accs).mean()
            print('eval epoch:%d step: %d loss:%f acc:%f' % (epoch, global_step, avg_loss, avg_acc))
            log_writer.add_scalar(tag="dev/loss", step=log_step, value=avg_loss)
            log_writer.add_scalar(tag="dev/acc", step=log_step, value=avg_acc)

            # 保存最佳模型
            if avg_acc>max_acc:
                max_acc = avg_acc
                print('saving the best_model...')
                paddle.save(model.state_dict(), 'best_model')
            model.train()

        # 反向传播
        loss.backward()

        # 使用优化器进行参数优化
        opt.step()

        # 清除梯度
        opt.clear_grad()

        # 全局步数加一
        global_step += 1

# 保存最终模型
paddle.save(model.state_dict(),'final_model')
```

    train epoch:0 step: 0 loss:2.471213 acc:0.187500
    eval epoch:0 step: 0 loss:2.471213 acc:0.187500
    saving the best_model...
    eval epoch:3 step: 500 loss:1.196341 acc:0.572917
    saving the best_model...
    eval epoch:6 step: 1000 loss:0.961887 acc:0.687500
    saving the best_model...


## 五、模型测试
* 使用保存的最佳模型进行测试


```python
# 反转字符表
label_dict_adv = {v: k for k, v in label_dict.items()}

# 输入计算题目
input_text = '12+40'

# 编码输入为ID
inputs = encoder(input_text, MAXLEN, label_dict)

# 转换输入为向量形式
inputs = np.array(inputs).reshape(-1, MAXLEN)
inputs = paddle.to_tensor(inputs)

# 加载模型
params_dict= paddle.load('best_model')
model.set_dict(params_dict)

# 设置为评估模式
model.eval()

# 模型推理
out = model(inputs)

# 结果转换
result = ''.join([label_dict_adv[_] for _ in np.argmax(out.numpy(), -1).reshape(-1)])

# 打印结果
print('the model answer: %s=%s' % (input_text, result))
print('the true answer: %s=%s' % (input_text, eval(input_text)))
```

    the model answer: 12+40=52 
    the true answer: 12+40=52


## 六、总结
* 你还可以通过变换网络结构，调整数据集，尝试不同的参数的方式来进一步提升本示例当中的数字加法的效果
* 同时，也可以尝试在其他的类似的任务中用飞桨来完成实际的实践
