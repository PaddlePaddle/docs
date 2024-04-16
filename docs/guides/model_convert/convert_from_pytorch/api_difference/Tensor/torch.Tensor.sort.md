## [ 仅参数名不一致 ]torch.Tensor.sort

### [torch.Tensor.sort](https://pytorch.org/docs/stable/generated/torch.Tensor.sort.html#torch-tensor-sort)

```python
torch.Tensor.sort(dim=- 1, descending=False)
```

### [paddle.sort](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sort_cn.html#sort)

```python
paddle.sort(x, axis=- 1, descending=False)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis         | 指定对输入 Tensor 进行运算的轴。默认值为-1, 仅参数名不一致。 |
| descending    |descending    | 指定算法排序的方向, 参数完全一致。     |
