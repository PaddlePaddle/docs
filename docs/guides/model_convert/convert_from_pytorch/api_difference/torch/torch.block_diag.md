## [输入参数类型不一致]torch.block_diag

### [torch.block_diag](https://pytorch.org/docs/stable/generated/torch.block_diag.html#torch-block-diag)

```python
torch.block_diag(*tensors)
```

### [paddle.block_diag](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/block_diag_cn.html)

```python
paddle.block_diag(inputs, name=None)
```

二者功能一致但参数类型不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                                         |
| -------- | ------------ | ------------------------------------------------------------ |
| *tensors | inputs       | 一组输入 Tensor，PyTorch 参数 tensors 为可变参数，Paddle 参数 inputs 为 list(Tensor) 或 tuple(Tensor) 的形式。 |

### 转写示例

#### *tensors：一组输入 Tensor

```python
# PyTorch 写法
torch.block_diag(x, y, z)

# Paddle 写法
paddle.block_diag([x, y, z])
```
