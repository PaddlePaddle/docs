## [部分参数不一致]torch.Tensor.svd

### [torch.Tensor.svd](https://pytorch.org/docs/1.13/generated/torch.Tensor.svd.html)

```python
    torch.Tensor.svd(some=True, compute_uv=True)
```

### [paddle.linalg.svd](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/svd_cn.html#svd)

```python
    paddle.linalg.svd(x, full_matrics=False, name=None)
```

### 不一致的参数
两者部分参数用法不同，具体如下：
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| some | full_matrices | 默认参数，默认一致|

### torch 多的参数
| 参数名        | 备注                                                                  |
| ------------- | -------------------------------------------------------------------- |
| compute_uv | 默认参数，默认一致。如果 compute_uv 为 False ，则返回的 U 和 V 将分别为形状为 (m, m) 和 (n ,n) 的零填充矩阵，并且与输入设备相同。当 compute_uv 为 False 时，参数 some 不起作用。 |

# 代码转写
```python
    # pytorch
    x = torch.randn(8, 2)
    U, S, V = x.svd(some = True, compute_uv=True)

    # paddle
    x = paddle.randn([8, 2])
    U, S, V = paddle.linalg.svd(x, full_matrices = True)
```
