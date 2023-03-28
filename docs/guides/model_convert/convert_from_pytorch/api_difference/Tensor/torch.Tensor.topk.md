## [ 仅参数名不一致 ] torch.Tensor.topk

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

| Pytorch | PaddlePaddle | 备注                             |
| ------- | ------------ | :------------------------------- |
| k       | k            | 表示前 k 个最大项。              |
| dim     | axis         | 表示排序的维度，仅参数名不一致。 |
| largest | largest      | True: 最大值， False: 最小值。   |
| sorted  | sorted       | 表示是否排序。                   |
