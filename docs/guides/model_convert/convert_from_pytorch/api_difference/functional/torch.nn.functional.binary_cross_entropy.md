## [ 参数不一致 ]torch.nn.functional.binary_cross_entropy

### [torch.nn.functional.binary\_cross\_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html)

```python
torch.nn.functional.binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.functional.binary\_cross\_entropy](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/binary_cross_entropy_cn.html#binary-cross-entropy)

```python
paddle.nn.functional.binary_cross_entropy(input, label, weight=None, reduction='mean', name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数不一致，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注 |
| ------------ | ------------ | -- |
| input        | input        | 输入 Tensor。 |
| target       | label        | 标签 Tensor。 |
| weight       | weight       | 手动指定每个 batch 二值交叉熵的权重。 |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| reduction    | reduction    | 指定应用于输出结果的计算方式。 |

### 转写示例

#### size_average：做 reduce 的方式
```python
# PyTorch 的 size_average、reduce 参数转为 Paddle 的 reduction 参数
if size_average is None:
    size_average = True
if reduce is None:
    reduce = True

if size_average and reduce:
    reduction = 'mean'
elif reduce:
    reduction = 'sum'
else:
    reduction = 'none'
```
