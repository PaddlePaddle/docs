## [torch 参数更多]torch.signbit

### [torch.signbit](https://pytorch.org/docs/stable/generated/torch.signbit.html#torch-signbit)

```python
torch.signbit(input, *, out=None)
```

### [paddle.signbit](https://github.com/PaddlePaddle/Paddle/blob/9ce3a54f456011c664c70fbcd318f2e1af0a7d81/python/paddle/tensor/math.py#L7175)

```python
paddle.signbit(x, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                           |
| ------- | ------------ | ---------------------------------------------- |
| input   | x            | 输入 Tensor，仅参数名不一致。                  |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.signbit([1., 2., 3., -1.], out=y)

# Paddle 写法
paddle.assign(paddle.signbit([1., 2., 3., -1.]), y)
```
