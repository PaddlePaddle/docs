## [ 返回参数类型不一致 ]torch.Tensor.equal

### [torch.Tensor.equal](https://pytorch.org/docs/stable/generated/torch.Tensor.equal.html?highlight=equal#torch.Tensor.equal)

```python
torch.Tensor.equal(other)
```

### [paddle.Tensor.equal_all](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#equal-all-y-name-none)

```python
paddle.Tensor.equal_all(y, name=None)
```

两者功能一致但返回参数类型不同，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                          |
| ------- | ------------ | ----------------------------- |
| other   | y            | 输入 Tensor，仅参数名不一致。 |
| 返回值   | 返回值        | PyTorch 返回 bool 类型，Paddle 返回 0-D bool Tensor，需要转写。|

### 转写示例
#### 返回值
```Python
# torch 中的写法
x.equal(y)

# paddle 中的写法
x.equal_all(y).item()
```
