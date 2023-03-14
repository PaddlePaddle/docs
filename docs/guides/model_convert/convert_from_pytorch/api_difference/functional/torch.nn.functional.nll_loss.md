## [ torch 参数更多 ]torch.nn.functional.nll_loss

### [torch.nn.functional.nll_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch-nn-functional-nll-loss)

```python
torch.nn.functional.nll_loss(input,
                    target,
                    weight=None,
                    size_average=None,
                    ignore_index=- 100,
                    reduce=None,
                    reduction='mean')
```

### [paddle.nn.functional.nll_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/nll_loss_cn.html#nll-loss)

```python
paddle.nn.functional.nll_loss(input,
                    label,
                    weight=None,
                    ignore_index=-100,
                    reduction='mean',
                    name=None)
```

其中 Pytorch 相⽐ Paddle ⽀持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | input         | 输入 Tensor                                     |
| target          | label         | 输入 Tensor 对应的标签值,仅参数名不一致。            |
| size_average          | -         | 已弃用                                      |
| weight          | weight  | 手动指定每个类别的权重                          |
| ignore_index          | ignore_index  |  指定一个忽略的标签值，此标签值不参与计算                   |
| reduce          | -         | 已弃用                                     |
| reduction          | reduction         | 表示应用于输出结果的规约方式，可选值有：'none', 'mean', 'sum'                         |

### 转写示例
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
