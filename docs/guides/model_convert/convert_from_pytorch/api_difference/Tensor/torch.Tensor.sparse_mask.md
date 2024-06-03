## [ 参数完全一致 ] torch.Tensor.sparse_mask

### [torch.Tensor.sparse_mask](https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_mask.html)

```python
torch.Tensor.sparse_mask(mask)
```

### [paddle.sparse.mask_as](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sparse/mask_as_cn.html)

```python
paddle.sparse.mask_as(x, mask, name=None)
```

两者功能一致，但调用方式不同，torch 通过 Tensor 类方法调用，而 paddle 是直接调用函数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                 |
| ---------- | ------------ | ------------------------------------ |
| -          | x            | 输入的 DenseTensor。                  |
| mask       | mask         | 掩码逻辑的 mask，参数完全一致。           |

### 转写示例

```python
# torch 调用 Tensor 类方法
x.sparse_mask(mask)

# paddle 直接调用函数
paddle.sparse.mask_as(x, mask)
```
