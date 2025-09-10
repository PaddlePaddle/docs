## [ torch 参数更多 ]torch.stack

### [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html#torch.stack)

```python
torch.stack(tensors,
            dim=0,
            *,
            out=None)
```

### [paddle.stack](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/stack_cn.html)

```python
paddle.stack(x,
             axis=0,
             name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  tensors  |  x     | 表示输入的 Tensor ，仅参数名不一致。  |
|  dim      |  axis  | 表示要堆叠的轴，仅参数名不一致。  |
|  out      |  -      | 表示输出的 Tensor，Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.stack(torch.tensor([3, 5]), out=y)

# Paddle 写法
paddle.assign(paddle.stack(paddle.to_tensor([3, 5])), y)
```
