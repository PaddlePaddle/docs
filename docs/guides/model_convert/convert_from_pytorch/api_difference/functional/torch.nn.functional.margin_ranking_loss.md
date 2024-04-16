## [torch 参数更多 ]torch.nn.functional.margin_ranking_loss

### [torch.nn.functional.margin_ranking_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.margin_ranking_loss.html?highlight=margin_ranking_loss#torch.nn.functional.margin_ranking_loss)

```python
torch.nn.functional.margin_ranking_loss(input1,
                                        input2,
                                        target,
                                        margin=0,
                                        size_average=None,
                                        reduce=None,
                                        reduction='mean')
```

### [paddle.nn.functional.margin_ranking_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/margin_ranking_loss_cn.html)

```python
paddle.nn.functional.margin_ranking_loss(input,
                                         other,
                                         label,
                                         margin=0.0,
                                         reduction='mean',
                                         name=None)
```

其中 PyTorch 相⽐ Paddle ⽀持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input1          | input         | 表示第一个输入的 Tensor ，仅参数名不一致。                                     |
| input2          | other         | 表示第二个输入的 Tensor ，仅参数名不一致。                                     |
| target          | label         | 表示训练数据的标签 Tensor ，仅参数名不一致。                                     |
| margin          | margin         | 表示用于加和的 margin 值 。                                     |
| size_average          | -         | 已弃用 。                                     |
| reduce          | -         | 已弃用 。                                     |
| reduction          | reduction         | 表示应用于输出结果的计算方式 。                                     |

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
