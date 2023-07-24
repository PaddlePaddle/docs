## [torch 参数更多]torch.Tensor.norm

### [torch.Tensor.norm](https://pytorch.org/docs/stable/generated/torch.Tensor.norm.html#torch.Tensor.norm)

```python
torch.Tensor.norm(p='fro', dim=None, keepdim=False, dtype=None)
```

### [paddle.Tensor.norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#norm-p-fro-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.norm(p='fro', axis=None, keepdim=False, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                          |
| ------- | ------------ | --------------------------------------------- |
| p       | p            | 范数(ord)的种类。                             |
| dim     | axis         | 使用范数计算的轴，仅参数名不一致。            |
| keepdim | keepdim      | 是否在输出的 Tensor 中保留和输入一样的维度。  |
| dtype   | -            | 输出数据类型，Paddle 无此参数，需要进行转写。 |

### 转写示例

#### dtype 参数：输出数据类型

```python
# Pytorch 写法
x.norm(dim=-1, dtype=torch.float32)

# Paddle 写法
y = x.astype('float32')
y.norm(dim=-1)
```
