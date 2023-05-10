## [参数不一致]torch.Tensor.svd

### [torch.Tensor.svd](https://pytorch.org/docs/1.13/generated/torch.Tensor.svd.html)

```python
torch.Tensor.svd(some=True, compute_uv=True)
```
### [paddle.linalg.svd](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/svd_cn.html#svd)

```python
paddle.linalg.svd(x, full_matrics=False, name=None)
```

Pytorch 的 `some` 参数与 Paddle 的 `full_matrices` 参数用法不同，且 Pytorch 有额外参数 `compute_uv`，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| some | full_matrices | 表示是否计算完整的 U 和 V 矩阵。默认时效果一致。 some 为 True 时， full_matrices 需设置为 False ，反之同理。 |
| compute_uv | - | 默认时效果一致。一般直接删除即可，无需转写。 compute_uv 为 False 时， paddle 暂无转写方式。|

### 转写示例
#### some ：表示是否计算完整的 U 和 V 矩阵。
```python
# pytorch
x = torch.randn(8, 2)
U, S, V = x.svd(some=True)

# paddle
x = paddle.randn([8, 2])
U, S, V = paddle.linalg.svd(x, full_matrices = False)
```
