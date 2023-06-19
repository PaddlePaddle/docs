# [torch 参数更多]torch.nn.functional.kl_div

### [torch.nn.functional.kl_div](https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html?highlight=kl_div#torch.nn.functional.kl_div)

```
torch.nn.functional.kl_div(input,
               target,
               size_average=None,
               reduce=None,
               reduction='mean',
               log_target=False)
```

### [paddle.nn.functional.kl_div](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/kl_div_cn.html)

```
paddle.nn.functional.kl_div(input,
                label,
                reduction='mean')
```

其中 PyTorch 相比 Paddle 支持更多的参数，具体如下：

| PyTorch      | PaddlePaddle | 备注                                                   |
| ------------ | ------------ | ------------------------------------------------------ |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。           |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。           |
| reduction    | reduction    | 表示对输出结果的计算方式。                             |
| log_target   | -            | 指定目标是否为 log 空间，Paddle 无此功能，暂无转写方式。 |

### 转写示例

#### size_average

```
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
