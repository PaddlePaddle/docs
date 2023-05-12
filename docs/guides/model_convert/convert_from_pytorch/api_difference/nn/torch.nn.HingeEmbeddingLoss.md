## [torch 参数更多]torch.nn.HingeEmbeddingLoss

### [torch.nn.HingeEmbeddingLoss](https://pytorch.org/docs/1.13/generated/torch.nn.HingeEmbeddingLoss.html#hingeembeddingloss)

```python
torch.nn.HingeEmbeddingLoss(margin=1.0,
                            size_average=None,
                            reduce=None,
                            reduction='mean')
```

### [paddle.nn.HingeEmbeddingLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/HingeEmbeddingLoss_cn.html#hingeembeddingloss)

```python
paddle.nn.HingeEmbeddingLoss(margin=1.0,
                             reduction='mean',
                             name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | Paddle    | 备注                                                         |
| ------------ | --------- | ------------------------------------------------------------ |
| margin       | margin    | 当 label 为-1 时，该值决定了小于 margin 的 input 才需要纳入 hinge embedding loss 的计算。 |
| size_average | -         | PyTorch 已弃用， Paddle 无此参数，需要转写。                 |
| reduce       | -         | PyTorch 已弃用， Paddle 无此参数，需要转写。                 |
| reduction    | reduction | 表示应用于输出结果的计算方式。                               |

### 转写示例

#### size_average

```python
# Pytorch 的 size_average、reduce 参数转为 Paddle 的 reduction 参数
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
