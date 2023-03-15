## [ torch 参数更多 ]torch.nn.functional.mse_loss

### [torch.nn.functional.mse_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html?highlight=mse_loss#torch.nn.functional.mse_loss)

```python
torch.nn.functional.mse_loss(input,
                             target,
                             size_average=None,
                             reduce=None,
                             reduction='mean')
```

### [paddle.nn.functional.mse_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/mse_loss_cn.html)

```python
paddle.nn.functional.mse_loss(input,
                              label,
                              reduction='mean',
                              name=None)
```

其中 Pytorch 相⽐ Paddle ⽀持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | input         | 表示预测的 Tensor 。                                     |
| target          | label         | 表示真实的 Tensor 。                                     |
| size_average          | -         | 已弃用 。                                     |
| reduce          | -         | 已弃用 。                                     |
| reduction          | reduction         | 表示应用于输出结果的计算方式 。                                     |

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
