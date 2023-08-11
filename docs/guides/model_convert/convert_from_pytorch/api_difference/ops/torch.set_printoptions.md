## [ torch 参数更多 ]torch.set_printoptions

### [torch.set_printoptions](https://pytorch.org/docs/stable/generated/torch.set_printoptions.html?highlight=torch+set_printoptions#torch.set_printoptions)

```python
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
```

### [paddle.set_printoptions][https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/set_printoptions_cn.html]

```python
paddle.set_printoptions(precision=None, threshold=None, edgeitems=None, sci_mode=None, linewidth=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                                         |
| --------- | ------------ | ------------------------------------------------------------ |
| precision | precision    | 浮点数的小数位数, Pytorch 默认值为 4，Paddle 默认为 8。      |
| threshold | threshold    | 打印的元素个数上限，默认值为 1000。                          |
| edgeitems | edgeitems    | 以缩略形式打印时左右两边的元素个数，默认值为 3。             |
| linewidth | linewidth    | 每行的字符数，默认值为 80。                                  |
| sci_mode  | sci_mode     | 是否以科学计数法打印，Pytorch 默认根据网络自动选择， Paddle 默认值为 False。 |
| profile   | -            | 预设风格，支持 `default`, `short`, `full`。 Paddle 无此参数， 需要转写。 |

### 转写示例

#### profile:预设风格，设置为 `default`。

```
# Pytorch 写法
torch.set_printoptions(profile='default')

# Paddle 写法
paddle.set_printoptions(precision=4, threshold=1000, edgeitems=3, linewidth=80)
```

#### profile:预设风格，设置为 `short`。

```
# Pytorch 写法
torch.set_printoptions(profile='short')

# Paddle 写法
paddle.set_printoptions(precision=2, threshold=1000, edgeitems=2, linewidth=80)
```

#### profile:预设风格，设置为 `full`。

```
# Pytorch 写法
torch.set_printoptions(profile='full')

# Paddle 写法
paddle.set_printoptions(precision=4, threshold=1000000, edgeitems=3, linewidth=80)
```
