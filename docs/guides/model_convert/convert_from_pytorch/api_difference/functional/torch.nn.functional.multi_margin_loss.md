## [ torch 参数更多 ]torch.nn.functional.multi_margin_loss

### [torch.nn.functional.multi\_margin\_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.multi_margin_loss.html)

```python
torch.nn.functional.multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.functional.multi\_margin\_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/multi_margin_loss_cn.html#multi-margin-loss)

```python
paddle.nn.functional.multi_margin_loss(input, label, p=1, margin=1.0, weight=None, reduction='mean', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注 |
| ------------ | ------------ | -- |
| input        | input        | 输入 Tensor，仅参数名不一致。 |
| target       | label        | 标签 Tensor，仅参数名不一致。 |
| p            | p            | 手动指定范数。|
| margin       | margin       | 手动指定间距。 |
| weight       | weight       | 手动指定每个 batch 二值交叉熵的权重。 |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| reduction    | reduction    | 指定应用于输出结果的计算方式。 |

### 转写示例

#### size_average、reduce
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
