# Python 文档示例代码书写规范

规范概要：

1. 第 1 节，显式的输出优于注释
2. 第 2 节，固定的输出优于随机
3. 第 3 节，明确的设备优于默认
4. 第 4 节，尝试去检查优于跳过

执行说明：

1. 规范在执行过程中，可能会发现现有规范未考虑到的方面，需要在实施过程中不断补充与完善，也请大家积极反馈意见。
2. 示例代码的执行限时为 `10` 秒，即，要求示例代码在 `10` 秒内执行完毕。如有特殊情况，如网络下载等情况，可以向 reviewer 提出增加 `TIMEOUT` 选项。
3. 如遇其他问题，如符合规范的示例代码无法进行检查等情况，请及时反馈 reviewer 进行确认。

## 1. 显式的输出优于注释

请尽量将用户可能关注的输出，如变量的值、`Tensor` 的 `shape` 等，书写或拷贝至示例中。

可以使用 `print` 输出结果，如：

``` python
>>> import paddle
>>> x = paddle.to_tensor([[1, 2], [3, 4]])
>>> y = paddle.to_tensor([[5, 6], [7, 8]])
>>> res = paddle.multiply(x, y)
>>> print(res)
Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
[[5 , 12],
 [21, 32]])
```

如无特殊情况，请勿使用 `#` 的注释方式提供输出值。

如：

``` python
>>> res = paddle.multiply(x, y) # shape=[2, 2]
```

``` python
>>> res = paddle.multiply(x, y)
>>> # [[5 , 12],
>>> #  [21, 32]]
```

等，都是不建议的输出方式。

另外，在书写或拷贝示例代码时，请注意以下几点：

- 输出中请 **不要** 留有空行。

    如：

    ``` python
    >>> import paddle
    >>> x = paddle.to_tensor([[1, 2], [3, 4]])
    >>> y = paddle.to_tensor([[5, 6], [7, 8]])
    >>> res = paddle.multiply(x, y)
    >>> print(res)
    Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
    [[5 , 12],

    [21, 32]])
    ```

    请改为：

    ``` python
    >>> import paddle
    >>> x = paddle.to_tensor([[1, 2], [3, 4]])
    >>> y = paddle.to_tensor([[5, 6], [7, 8]])
    >>> res = paddle.multiply(x, y)
    >>> print(res)
    Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
    [[5 , 12],
     [21, 32]])
    ```

    对于代码中的多行或复合语句，也 **不要** 留有空行。

    如：

    ``` python
    >>> class Mnist(nn.Layer):
    ...     def __init__(self):

    ...         super().__init__()
    ...

    ```

    请改为：

    ``` python
    >>> class Mnist(nn.Layer):
    ...     def __init__(self):
    ...
    ...         super().__init__()
    ...
    ```

    或者：

    ``` python
    >>> class Mnist(nn.Layer):
    ...     def __init__(self):
    ...         super().__init__()
    ...
    ```

- 输出请统一 **左对齐** 其上方的 `>>> ` 或 `... `。

    如：

    ``` python
    >>> import paddle
    >>> x = paddle.to_tensor([[1, 2], [3, 4]])
    >>> y = paddle.to_tensor([[5, 6], [7, 8]])
    >>> res = paddle.multiply(x, y)
    >>> print(res)
        Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
        [[5 , 12],
        [21, 32]])
    ```

    请改为：

    ``` python
    >>> import paddle
    >>> x = paddle.to_tensor([[1, 2], [3, 4]])
    >>> y = paddle.to_tensor([[5, 6], [7, 8]])
    >>> res = paddle.multiply(x, y)
    >>> print(res)
    Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
    [[5 , 12],
     [21, 32]])
    ```

- 适度的空格用于美化输出是允许的。

    如：

    ``` python
    >>> import paddle
    >>> x = paddle.to_tensor([[1, 2], [3, 4]])
    >>> y = paddle.to_tensor([[5, 6], [7, 8]])
    >>> res = paddle.multiply(x, y)
    >>> print(res)
    Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
    [[5 , 12],
    [21, 32]])
    ```

    可以对齐其中的方括号为：


    ``` python
    >>> import paddle
    >>> x = paddle.to_tensor([[1, 2], [3, 4]])
    >>> y = paddle.to_tensor([[5, 6], [7, 8]])
    >>> res = paddle.multiply(x, y)
    >>> print(res)
    Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
    [[5 , 12],
     [21, 32]])
    ```

- 对于多行的输出，可以使用 `...` 作为通配符使用。

    如：

    ``` python
    >>> sampler = MySampler(data_source=RandomDataset(100))
    >>> for index in sampler:
    ...     print(index)
    0
    1
    2
    ...
    99
    ```

## 2. 固定的输出优于随机

请尽量保证输出为固定值，对于示例中的随机情况，请设置随机种子。

如：

``` python
>>> import paddle
>>> paddle.seed(2023)
>>> data = paddle.rand(shape=[2, 3])
>>> print(data)
Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
[[0.86583614, 0.52014720, 0.25960937],
 [0.90525323, 0.42400089, 0.40641287]])
```

如果示例中涉及 `Python`、`Numpy` 随机数等情况，也需要进行随机种子的设置。

## 3. 明确的设备优于默认

由于 `Tensor` 在 `CPU`、`GPU` 等设备上面的行为可能不同，请写明具体的设备需求。

如：

``` text
>>> # doctest: +REQUIRES(env:GPU)
>>> import paddle
>>> paddle.device.set_device('gpu')
>>> count = paddle.device.cuda.device_count()
>>> print(count)
1
```

其中：

- 第 `1` 行是 `doctest` 检查指令。
- 第 `3` 行是设备的设置。

CI 中默认使用 `CPU` 作为检查环境，因此，如无特殊情况，可以 **不用** 写明 `CPU` 环境，如：

``` python
>>> paddle.device.set_device('cpu') # 可以不写
```

另外，对于有 `GPU`、`XPU` 等设备需求的情况，需要在代码开头添加 `doctest` 指令，如：

- 需要 `GPU` 环境

    ``` text
    >>> # doctest: +REQUIRES(env:GPU)
    >>> ...
    ```

- 需要 `XPU` 环境

    ``` text
    >>> # doctest: +REQUIRES(env:XPU)
    >>> ...
    ```

- 需要 `GPU`、`XPU` 等多个环境

    ``` text
    >>> # doctest: +REQUIRES(env:GPU, env:XPU)
    >>> ...
    ```

请注意这里的 **大小写** ，其中 `doctest` 为小写，`REQUIRES` 为大写，`env` 为小写，`GPU` 为大写。

## 4. 尝试去检查优于跳过

示例代码的检查可以保证其正确性，但并不是所有代码均能够正常或正确的在 CI 环境中运行。

如：

- 依赖外部资源才能运行，如 `image`、`audio` 等文件。
- 代码的随机性由系统决定，如 [os.walk](https://docs.python.org/3/library/os.html#os.walk) 等方法。
- 代码不能在 CI 中正常运行，如 `inspect.getsourcelines` 会抛出 `raise OSError('could not get source code')`

等情况，此时，可以使用 `doctest` 的 `SKIP` 指令跳过检查，如：

``` text
>>> # doctest: +SKIP('file not exist')
>>> with open('cat.jpg') as f:
...     im = load_image_bytes(f.read())
```

`SKIP` 指令还可以成对使用，如：
``` text
>>> # doctest: +SKIP('file not exist')
>>> with open('cat.jpg') as f:
...     im = load_image_bytes(f.read())
>>> # doctest: -SKIP
>>> x = paddle.to_tensor([[1, 2], [3, 4]])
>>> ...
```

其中，

- `+SKIP` 表示后面的代码要跳过，`-SKIP` 表示恢复检查。
- `+SKIP` 可以加上说明，如 `+SKIP('file not exist')`


## TIMEOUT 选项

示例代码检查的默认限制时间为 `10` 秒，但有些情况下代码无法及时完成运行，如，需要通过网络下载较大的模型或数据集。如果出现此类情况，开发者可以与 reviewer 协商增加 `TIMEOUT` 选项，如：

``` python
>>> from paddle.vision.datasets import VOC2012
>>> voc2012 = VOC2012()
```

此段示例代码需要通过网络进行下载，运行时间超过了限制的执行时间 `10` 秒，此时，在 reviewer 同意的情况下，可以添加 `TIMEOUT` 选项，修改为：

``` text
>>> # doctest: +TIMEOUT(60)
>>> from paddle.vision.datasets import VOC2012
>>> voc2012 = VOC2012()
```

此时，此段示例代码的执行限时被修改为 `60` 秒。

`TIMEOUT` 的具体时长可根据实际情况进行修改。


## SOLO 选项

`SOLO` 多指音乐中的 `独奏`，在示例代码的写作中，用以指代 `Main Process Separately` ，即，指示示例检查工具使用 `主进程单独` 运行此示例代码。

示例检查工具使用了 `python` 中的 `multiprocessing` 模块，虽然 `multiprocessing` 多应用于多进程加速，但，在示例检查中，主要用作隔离各个示例代码的检查环境，这有几个原因：

- `Paddle` 是一个庞大且复杂的框架，涉及较多全局变量，以及动态图与静态图转换等操作，客观上要求每次示例代码的运行保持干净的初始化状态

- 示例代码检查与单元测试不同，一般不会进行复杂的运行前后处理（如 `setUp`, `tearDown` ），这就导致了单个示例运行之后，可能污染检查环境

- `Paddle` 内部的一些接口，如异步 `io` 等操作，也会涉及到多进程操作，这就要求示例检查工具不能够影响 `Paddle` 自身的进程管理

基于此，示例检查工具遵循以下设计：

- 示例检查工具在每次运行检查前，都会单独创建进程，并在进程中 `import paddle`，以初始化环境

- 示例检查工具使用 `python` 的 `multiprocessing` 开辟进程，基于安全性的考虑（[Contexts and start methods](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)），使用 `spawn` 的方式

- 由于多进程嵌套可能存在问题，如示例中使用异步 `io` 又开辟多个进程，特此引入 `SOLO` 选项，指示示例检查工具，此示例代码需要在主进程中单独运行

以一个简单的多进程代码为例：

``` python

>>> import multiprocessing
>>> p = multiprocessing.Process(
...     target=lambda a, b: a + b,
...     args=(1, 2),
... )
>>> p.start()
>>> p.join()

```

在 reviewer 同意的情况下，可以添加 `SOLO` 选项，修改为：

``` text

>>> # doctest: +SOLO('can not use add in multiprocessing')
>>> import multiprocessing
>>> p = multiprocessing.Process(
...     target=lambda a, b: a + b,
...     args=(1, 2),
... )
>>> p.start()
>>> p.join()

```

此时，示例单独在主进程中运行检查，并能正常通过。

**注意：** 使用 `SOLO` 选项时，由于主进程环境没有特殊的运行前和运行后处理，因此，如果一次运行多个 `SOLO` 的示例，需要事先确认各个示例间不会相会影响。


## 参考资料：
- [「将 xdoctest 引入到飞桨框架工作流中」 RFC](https://github.com/PaddlePaddle/community/blob/master/rfcs/Docs/%E5%B0%86%20xdoctest%20%E5%BC%95%E5%85%A5%E5%88%B0%E9%A3%9E%E6%A1%A8%E6%A1%86%E6%9E%B6%E5%B7%A5%E4%BD%9C%E6%B5%81%E4%B8%AD.md)
- [「将 xdoctest 引入到飞桨框架工作流中」 详细设计](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/sampcd_processor_readme.md)
- [「doctest」官方文档 ](https://docs.python.org/3/library/doctest.html#module-doctest)
- [ 「xdoctest」官方文档 ](https://xdoctest.readthedocs.io/en/latest/)
