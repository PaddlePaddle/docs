## [ torch 参数更多 ]torch.Tensor.gather

### [torch.gather](https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather)

```python
torch.gather(input, dim, index, *, sparse_grad=False, out=None)
```

### [paddle.gather](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/gather_cn.html#gather)

```python
paddle.gather(x, index, axis=None, name=None)
```

两者功能一致且参数用法一致，torch 参数更多，具体如下：

### 参数映射

| PyTorch                        | PaddlePaddle             | 备注                                              |
| ------------------------------ | ------------------------ | ------------------------------------------------- |
| <center> input </center>       | <center> x </center>     | 输入 Tensor，仅参数名不同。                       |
| <center> dim </center>         | <center> index </center> | 索引 Tensor，仅参数名不同。                       |
| <center> index </center>       | <center> axis </center>  | 指定 index 获取输入的维度，仅参数名不同。         |
| <center> sparse_grad </center> | <center> - </center>     | pytorch：如果是 `True`，输入则是一个稀疏 Tensor。 |
