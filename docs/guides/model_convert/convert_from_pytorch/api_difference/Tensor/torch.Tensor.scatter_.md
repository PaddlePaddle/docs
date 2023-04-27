## [仅参数名不一致]torch.Tensor.scatter_

### [torch.Tensor.scatter_](https://pytorch.org/docs/1.13/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_)

```python
torch.Tensor.scatter_(dim,
                      index,
                      src,
                      reduce=None)
```

### [paddle.Tensor.put_along_axis_](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/put_along_axis__cn.html)

```python
paddle.Tensor.put_along_axis_(indices,
                              value,
                              axis,
                              reduce)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis        | 表示在哪一个维度 scatter ，仅参数名不一致。 |
| index         | indices     | 表示输入的索引张量，仅参数名不一致。                   |
| src           | values      | 表示需要插入的值，仅参数名不一致。                   |
| reduce        | reduce      | 表示对输出 Tensor 的计算方式。  |
