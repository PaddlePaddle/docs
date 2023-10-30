## [ 仅 paddle 参数更多 ] torch.nansum

### [torch.nansum](https://pytorch.org/docs/stable/generated/torch.nansum.html)

```python
torch.nansum(input, *, dtype=None)
```

### [paddle.nansum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nansum_cn.html)

```python
paddle.nansum(x, axis=None, dtype=None, keepdim=False, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                                      |
| ----------- | ------------ | ----------------------------------------------------------------------------------------- |
| input       | x            | 输入的 Tensor，数据类型为：float16、float32、float64、int32 或 int64。仅参数名不一致。       |
| dtype       | dtype        | 输出变量的数据类型。若参数为空，则输出变量的数据类型和输入变量相同，默认值为 None。            |
