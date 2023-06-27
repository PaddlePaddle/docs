## [ 仅参数名不一致 ]torch.cdist

### [torch.cdist](https://pytorch.org/docs/stable/generated/torch.cdist.html#torch.cdist)

```python
torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
```
### [paddle.cdist]()

```python
paddle.cdist(x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary', name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> x1 </font> | <font color='red'> x </font> | 表示第一个输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> x2 </font> | <font color='red'> y </font> | 表示第二个输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> p </font> | <font color='red'> p </font> | 表示 p-norm 距离的值。  |
| <font color='red'> compute_mode </font> | <font color='red'> compute_mode </font> |表示计算模式。  |
