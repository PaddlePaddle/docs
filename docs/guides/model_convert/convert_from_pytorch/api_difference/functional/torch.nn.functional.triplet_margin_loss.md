## [torch 参数更多 ] torch.nn.functional.triplet_margin_loss

### [torch.nn.functional.triplet_margin_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.triplet_margin_loss.html?highlight=triplet_margin_loss#torch.nn.functional.triplet_margin_loss)

```python
torch.nn.functional.triplet_margin_loss(anchor,
                positive,
                negative,
                margin=1.0,
                p=2,
                eps=1e-06,
                swap=False,
                size_average=None,
                reduce=None,
                reduction='mean')
```

### [paddle.nn.functional.triplet_margin_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/triplet_margin_loss_cn.html)

```python
paddle.nn.functional.triplet_margin_loss(input,
                positive,
                negative,
                margin: float = 1.0,
                p: float = 2.0,
                epsilon: float = 1e-6,
                swap: bool = False,
                reduction: str = 'mean',
                name: str = None)
```

其中 Pytorch 相⽐ Paddle ⽀持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| anchor          | input         | 输入 Tensor，仅参数名不一致。                        |
| positive          | positive         | 输入正样本                                 |
| negative          | negative         | 输入负样本                                     |
| margin          | margin         |  手动指定间距                                  |
| p          | p         | 指定范数                                 |
| eps          | epsilon         | 防止除数为零的常数                                  |
| swap          | swap         | 是否进行交换                                  |
| size_average          | -         | 已弃用                                      |
| reduce          | -         | 已弃用                                     |
| reduction          | reduction         | 表示应用于输出结果的规约方式，可选值有：'none', 'mean', 'sum'             |

### 转写示例
#### size_average
```python
# Pytorch 的 size_average、 reduce 参数转为 Paddle 的 reduction 参数
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
