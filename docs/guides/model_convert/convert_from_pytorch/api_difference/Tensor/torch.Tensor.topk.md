## torch.Tensor.topk

### [torch.Tensor.topk](https://pytorch.org/docs/1.13/generated/torch.Tensor.topk.html#torch.Tensor.topk)

```
torch.Tensor.topk(k, dim=None, largest=True, sorted=True)
```

### [paddle.Tensor.topk](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#topk-k-axis-none-largest-true-sorted-true-name-none)

```
paddle.Tensor.topk(k, axis=None, largest=True, sorted=True, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| Pytorch | PaddlePaddle | 备注                      |
| ------- | ------------ | :------------------------ |
| k       | k            | 前k个最大项               |
| dim     | axis         | 排序的维度                |
| largest | largest      | True:最大值，False:最小值 |
| sorted  | sorted       | 是否排序                  |

