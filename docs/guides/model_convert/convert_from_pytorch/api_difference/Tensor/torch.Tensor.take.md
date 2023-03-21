## torch.Tensor.take

### [torch.Tensor.take](https://pytorch.org/docs/1.13/generated/torch.Tensor.take.html#torch.Tensor.take)

```
torch.Tensor.take(indices)
```

### [paddle.Tensor.take](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#take-index-mode-raise-name-none)

```
paddle.Tensor.take(index, mode='raise', name=None)
```

两者功能一致，仅参数名不一致，paddlepaddle有多余参数mode,具体如下：

### 参数映射

| Pytorch | PaddlePaddle | 备注                        |
| ------- | ------------ | --------------------------- |
| indices | index        | 输入tensor的索引            |
|         | mode         | 指定索引越界的 3 种处理方式 |

注：

三种mode

- `mode='raise'`，若索引越界，通过最后调用的 `paddle.index_select` 抛出错误 （默认）；
- `mode='wrap'`，通过取余约束越界的 indices；
- `mode='clip'`，通过 `paddle.clip` 将两端超出范围的索引约束到 [0, max_index-1]。
