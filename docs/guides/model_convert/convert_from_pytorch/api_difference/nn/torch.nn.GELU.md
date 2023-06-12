## [参数不一致]torch.nn.GELU

### [torch.nn.GELU](https://pytorch.org/docs/1.13/generated/torch.nn.GELU.html#torch.nn.GELU)

```python
torch.nn.GELU(approximate='none')
```

### [paddle.nn.GELU](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/GELU_cn.html)

```python
paddle.nn.GELU(approximate=False, name=None)
```

其中功能一致, 输入参数用法不一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                                 |
| ----------- | ------------ | ------------------------------------------------------------------------------------ |
| approximate | approximate  | 是否使用近似计算，PyTorch 取值 none 和 tanh，Paddle 取值为 bool 类型，需要进行转写。 |

### 转写示例

#### approximate 参数：是否使用近似计算

```python
# PyTorch 写法:
m = torch.nn.GELU(approximate='tanh')
x = torch.tensor([[-1, 0.5],[1, 1.5]])
y = m(x)

# Paddle 写法:
m = paddle.nn.GELU(approximate=True)
x = paddle.to_tensor([[-1, 0.5],[1, 1.5]])
y = m(x)
```
