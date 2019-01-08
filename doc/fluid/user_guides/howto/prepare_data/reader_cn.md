```eval_rst
.. _user_guide_reader:
```

# Python Reader
在模型训练和预测阶段，PaddlePaddle程序需要读取训练或预测数据。为了帮助用户编写数据读取的代码，我们提供了如下接口：

- *reader*: 用于读取数据的函数，数据可来自于文件、网络、随机数生成器等，函数每次返回一个数据项。
- *reader creator*: 接受一个或多个reader作为参数、返回一个新reader的函数。
- *reader decorator*: 一个函数，接受一个或多个reader，并返回一个reader。
- *batch reader*: 用于读取数据的函数，数据可来自于文件、网络、随机数生成器等，函数每次返回一个batch大小的数据项。

此外，还提供了将reader转换为batch reader的函数，会频繁用到reader creator和reader decorator。

## Data Reader 接口
Data reader不一定要求为读取和遍历数据项的函数。它可以是返回iterable对象（即可以用于`for x in iterable`的任意对象）的任意不带参数的函数：

```
iterable = data_reader()
```

Iterable对象应产生单项或tuple形式的数据，而不是一个mini batch的数据。产生的数据项应在[支持的类型](./feeding_data.html#fluid) 中，例如float32,int类型的numpy一维矩阵，int类型的列表等。

以下是实现单项数据reader creator的示例：

```python
def reader_creator_random_image(width, height):
    def reader():
        while True:
            yield numpy.random.uniform(-1, 1, size=width*height)
    return reader
```

以下是实现多项数据reader creator的示例：

```python
def reader_creator_random_image_and_label(width, height, label):
    def reader():
        while True:
            yield numpy.random.uniform(-1, 1, size=width*height), label
    return reader
```

## Batch Reader 接口
*Batch reader*可以是返回iterable对象（即可以用于`for x in iterable`的任意对象）的任意不带参数的函数。Iterable的输出应为一个batch（list）的数据项。list中的每个数据项均为一个tuple元组。

这里是一些有效输出：

```python
# 三个数据项组成一个mini batch。每个数据项有三列，每列数据项为1。
[(1, 1, 1),
(2, 2, 2),
(3, 3, 3)]

# 三个数据项组成一个mini batch。每个数据项是一个列表（单列）。
[([1,1,1],),
([2,2,2],),
([3,3,3],)]
```

请注意列表里的每个项必须为tuple，下面是一个无效输出：
```python
 # 错误, [1,1,1]需在一个tuple内: ([1,1,1],).
 # 否则产生歧义，[1,1,1]是否表示数据[1, 1, 1]整体作为单一列。
 # 或者数据的三列，每一列为1。
[[1,1,1],
[2,2,2],
[3,3,3]]
```

很容易将reader转换成batch reader：

```python
mnist_train = paddle.dataset.mnist.train()
mnist_train_batch_reader = paddle.batch(mnist_train, 128)
```

也可以直接创建一个自定义batch reader：

```python
def custom_batch_reader():
    while True:
        batch = []
        for i in xrange(128):
            batch.append((numpy.random.uniform(-1, 1, 28*28),)) # note that it's a tuple being appended.
        yield batch

mnist_random_image_batch_reader = custom_batch_reader
```

## 使用
以下是我们如何用PaddlePaddle的reader：

batch reader是从数据项到数据层（data layer）的映射，batch_size和总pass数通过以下方式传给`paddle.train`：

```python
# 创建两个数据层：
image_layer = paddle.layer.data("image", ...)
label_layer = paddle.layer.data("label", ...)

# ...
batch_reader = paddle.batch(paddle.dataset.mnist.train(), 128)
paddle.train(batch_reader, {"image":0, "label":1}, 128, 10, ...)
```

## Data Reader装饰器
*Data reader decorator*接收一个或多个reader对象作为参数，返回一个新的reader对象。它类似于[python decorator](https://wiki.python.org/moin/PythonDecorators) ，但在语法上不需要写`@`。

我们对data reader接口有严格限制（无参数并返回单个数据项），data reader可灵活地搭配data reader decorators使用。以下是一些示例：

### 预取回数据（缓存数据）
由于读数据需要一些时间，而没有数据无法进行训练，因此一般而言数据预读取会是一个很好的方法。

用`paddle.reader.buffered`预读取数据：

```python
buffered_reader = paddle.reader.buffered(paddle.dataset.mnist.train(), 100)
```

`buffered_reader`将尝试缓存（预读取）`100`个数据项。

### 组成多个Data Reader
例如，如果我们想用实际图像源(也就是复用mnist数据集),和随机图像源作为[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)的输入。

我们可以参照如下：

```python
def reader_creator_random_image(width, height):
    def reader():
        while True:
            yield numpy.random.uniform(-1, 1, size=width*height)
    return reader

def reader_creator_bool(t):
    def reader:
        while True:
            yield t
    return reader

true_reader = reader_creator_bool(True)
false_reader = reader_creator_bool(False)

reader = paddle.reader.compose(paddle.dataset.mnist.train(), data_reader_creator_random_image(20, 20), true_reader, false_reader)
# 跳过1因为paddle.dataset.mnist.train()为每个数据项生成两个项。
# 并且这里我们暂时不考虑第二项。
paddle.train(paddle.batch(reader, 128), {"true_image":0, "fake_image": 2, "true_label": 3, "false_label": 4}, ...)
```

### 随机排序
给定大小为`n`的随机排序缓存， `paddle.reader.shuffle`返回一个data reader ，缓存`n`个数据项，并在读取一个数据项前进行随机排序。

示例：
```python
reader = paddle.reader.shuffle(paddle.dataset.mnist.train(), 512)
```

## Q & A

### 为什么一个reader只返回单项而不是mini batch？

返回单项，可以更容易地复用已有的data reader，例如如果一个已有的reader返回3项而不是一个单项，这样训练代码会更复杂，因为需要处理像batch_size为2这样的例子。

我们提供一个函数来将一个单项reader转换成一个batch reader。

### 为什么需要一个batch raeder，在训练过程中给出reader和batch_size参数这样不足够吗？

在大多数情况下，在训练方法中给出reader和batch_size参数是足够的。但有时用户想自定义mini batch里数据项的顺序，或者动态改变batch_size。在这些情况下用batch reader会非常高效有用。

### 为什么用字典而不是列表进行映射？

使用字典(`{"image":0, "label":1}`)而不是列表`["image", "label"]`)有利于用户易于复用数据项，例如使用`{"image_a":0, "image_b":0, "label":1}`，或者甚至跳过数据项，例如使用`{"image_a":0, "label":2}`。


### 如何创建一个自定义data reader？
```python
def image_reader_creator(image_path, label_path, n):
    def reader():
        f = open(image_path)
        l = open(label_path)
        images = numpy.fromfile(
            f, 'ubyte', count=n * 28 * 28).reshape((n, 28 * 28)).astype('float32')
        images = images / 255.0 * 2.0 - 1.0
        labels = numpy.fromfile(l, 'ubyte', count=n).astype("int")
        for i in xrange(n):
            yield images[i, :], labels[i] # a single entry of data is created each time
        f.close()
        l.close()
    return reader

# images_reader_creator创建一个reader
reader = image_reader_creator("/path/to/image_file", "/path/to/label_file", 1024)
paddle.train(paddle.batch(reader, 128), {"image":0, "label":1}, ...)
```

### `paddle.train`实现原理
实现`paddle.train`的示例如下：

```python
def train(batch_reader, mapping, batch_size, total_pass):
    for pass_idx in range(total_pass):
        for mini_batch in batch_reader(): # this loop will never end in online learning.
            do_forward_backward(mini_batch, mapping)
```
