## [参数不一致]torch.Tensor.svd

### [torch.Tensor.svd](https://pytorch.org/docs/1.13/generated/torch.Tensor.svd.html)

```python
    torch.Tensor.svd(some=True, compute_uv=True)
```

### [paddle.linalg.svd](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/svd_cn.html#svd)

```python
    paddle.linalg.svd(x, full_matrics=False, name=None)
```

### 参数映射
两者部分参数用法不同，具体如下：
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| some | full_matrices | 表示是否计算完整的 U 和 V 矩阵。默认时效果一致。 some 为 True 时， full_matrices 需设置为 False ，反之同理。 |
| compute_uv | - | 如果 compute_uv 为 False ，则返回的 U 和 V 将分别为形状为 (m, m) 和 (n ,n) 的零填充矩阵，并且与输入设备相同。当 compute_uv 为 False 时，参数 some 不起作用。一般直接删除即可，无需转写。 |

<!-- ### torch 多的参数
| 参数名        | 备注                                                                  |
| ------------- | -------------------------------------------------------------------- |
| compute_uv | 默认参数，默认一致。如果 compute_uv 为 False ，则返回的 U 和 V 将分别为形状为 (m, m) 和 (n ,n) 的零填充矩阵，并且与输入设备相同。当 compute_uv 为 False 时，参数 some 不起作用。 | -->

<!-- ### 转写示例
#### torch 的 some 指定为 False 时， paddle 的 full_matrices 需指定为 True 。
```python
    # pytorch
    x = torch.randn(8, 2)
    U, S, V = x.svd(some = False)

    # paddle
    x = paddle.randn([8, 2])
    U, S, V = paddle.linalg.svd(x, full_matrices = True) -->
```
