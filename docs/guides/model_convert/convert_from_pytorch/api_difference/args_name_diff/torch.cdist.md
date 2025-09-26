## [ 仅参数名不一致 ]torch.cdist

### [torch.cdist](https://pytorch.org/docs/stable/generated/torch.cdist.html#torch.cdist)

```python
torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
```

### [paddle.cdist](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/cdist_cn.html#cdist)

```python
paddle.cdist(x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary', name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  x1  |  x  | 表示第一个输入的 Tensor ，仅参数名不一致。  |
|  x2 |  y  | 表示第二个输入的 Tensor ，仅参数名不一致。  |
|  p  |  p  | 计算每个向量对之间的 p 范数距离的值。默认值为 2.0  |
|  compute_mode  |  compute_mode  |表示计算模式。  |
