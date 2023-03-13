## torch.nn.functional.l1_loss

### [torch.nn.functional.l1_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.l1_loss.html?highlight=l1_loss#torch.nn.functional.l1_loss)

```python
torch.nn.functional.l1_loss(input, target, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.functional.l1_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/l1_loss_cn.html)

```python
paddle.nn.functional.l1_loss(input, label, reduction='mean', name=None)
```

两者功能一致，torch 参数多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| target        | label        | 表示输入的 Tensor                                       |
| size_average  | -            | 已废弃，和 reduce 组合决定损失计算方式                        |
| reduce        | -            | 已废弃，和 size_average 组合决定损失计算方式                  |

### 转写示例

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
