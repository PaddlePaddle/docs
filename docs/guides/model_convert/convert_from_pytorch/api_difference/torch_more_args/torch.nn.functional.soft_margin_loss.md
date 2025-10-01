## [ torch 参数更多 ]torch.nn.functional.soft_margin_loss

### [torch.nn.functional.soft_margin_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.soft_margin_loss.html?highlight=soft_margin_loss#torch.nn.functional.soft_margin_loss)

```python
torch.nn.functional.soft_margin_loss(input,
                             target,
                             size_average=None,
                             reduce=None,
                             reduction='mean')
```

### [paddle.nn.functional.soft_margin_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/soft_margin_loss_cn.html)

```python
paddle.nn.functional.soft_margin_loss(input,
                              label,
                              reduction='mean',
                              name=None)
```

其中 PyTorch 相⽐ Paddle ⽀持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | input         | 输入 Tensor 。                                    |
| target          | label         | 输入 Tensor 对应的标签值，仅参数名不一致。               |
| size_average          | -         | 已弃用                                      |
| reduce          | -         | 已弃用                                     |
| reduction          | reduction         | 表示应用于输出结果的规约方式，可选值有：'none', 'mean', 'sum'。        |

### 转写示例
#### size_average
```python
# PyTorch 的 size_average、 reduce 参数转为 Paddle 的 reduction 参数
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
