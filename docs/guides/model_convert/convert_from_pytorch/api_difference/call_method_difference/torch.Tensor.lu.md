## [ 参数完全一致 ]torch.Tensor.lu

### [torch.Tensor.lu](https://pytorch.org/docs/stable/generated/torch.Tensor.lu.html)

```python
torch.Tensor.lu(pivot=True, get_infos=False)
```

### [paddle.Tensor.lu]()

```python
paddle.Tensor.lu(pivot=True, get_infos=False, name=None)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注 |
| --------- | ------------ | -- |
| pivot     | pivot        | LU 分解时是否进行旋转。 |
| get_infos | get_infos    | 是否返回分解状态信息，若为 True，则返回分解状态 Tensor，否则不返回。默认 False。 |
