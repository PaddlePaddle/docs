## [ 输入参数用法不一致 ]torch.tensor_split
### [torch.tensor_split](https://pytorch.org/docs/stable/generated/torch.tensor_split.html)

```python
torch.tensor_split(input, indices_or_sections, dim=0)
```

### [paddle.tensor_split](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/tensor_split_cn.html)

```python
paddle.tensor_split(x, num_or_indices, axis=0, name=None)
```

其中 Paddle的num_or_indices 与 PyTorch 的indices_or_sections用法不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                        |
| indices_or_sections           | num_or_indices         | indices_or_sections必须分开成indices和sections使用，且均可作为位置参数或关键字参数，num_or_indices不可分开。                          |
| dim           | axis         | 表示需要分割的维度，仅参数名不一致。                          |
