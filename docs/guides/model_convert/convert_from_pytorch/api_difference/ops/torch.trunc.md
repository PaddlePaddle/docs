## [ torch 参数更多 ]torch.trunc
### [torch.trunc](https://pytorch.org/docs/stable/generated/torch.trunc.html?highlight=torch+trunc#torch.trunc)

```python
torch.trunc(input,
          *,
          out=None)
```

### [paddle.trunc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/trunc_cn.html)

```python
paddle.trunc(input,
             name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|   input       |  input  | 表示输入的 Tensor。  |
|  out  | - |  表示输出的 Tensor ， Paddle 无此参数，需要转写。    |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.trunc(input, out=y)

# Paddle 写法
paddle.assign(paddle.trunc(input), y)
```
