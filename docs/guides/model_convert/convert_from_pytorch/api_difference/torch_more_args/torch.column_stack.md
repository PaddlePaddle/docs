## [ torch 参数更多 ]torch.column_stack
### [torch.column\_stack](https://docs.pytorch.org/docs/stable/generated/torch.column_stack.html#torch.column_stack)
```python
torch.column_stack(tensors, *, out=None)
```

### [paddle.column\_stack](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/column_stack_cn.html#paddle.column_stack)
```python
paddle.column_stack(x, name=None)
```

其中 Paddle 相比 PyTorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| tensors       | x            | 表示输入的 Tensor ，仅参数名不一致。                      |
| out           | -            | 表示输出的 Tensor ，Paddle 无此参数，需要转写。          |

### 转写示例
#### out 参数转写
```python
# PyTorch 写法
out = torch.tensor([[4, 5, 6], [1, 2, 3]])
result = torch.column_stack((a,b), out=out)

# Paddle 写法
out = paddle.tensor([[4, 5, 6], [1, 2, 3]])
result = paddle.assign(paddle.column_stack(x=(a, b)), output=out)

```
