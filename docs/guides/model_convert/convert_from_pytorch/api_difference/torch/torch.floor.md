## [torch 参数更多]torch.floor
### [torch.floor](https://pytorch.org/docs/stable/generated/torch.floor.html?highlight=torch+floor#torch.floor)

```python
torch.floor(input,
            *,
            out=None)
```

### [paddle.floor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/floor_cn.html#floor)

```python
paddle.floor(x,
             name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
|  out  | - |  表示输出的 Tensor ， Paddle 无此参数，需要转写。    |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.floor(torch.tensor([-0.4, -0.2, 0.1, 0.3]), out=y)

# Paddle 写法
paddle.assign(paddle.floor(paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])), y)
```
