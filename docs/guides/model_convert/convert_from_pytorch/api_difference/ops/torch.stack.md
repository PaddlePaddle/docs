## [torch 参数更多 ]torch.stack
### [torch.stack](https://pytorch.org/docs/1.13/generated/torch.stack.html#torch.stack)

```python
torch.stack(tensors,
            dim=0,
            *,
            out=None)
```

### [paddle.stack](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/stack_cn.html)

```python
paddle.stack(x,
             axis=0,
             name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  tensors  |  x     | 表示输入的 Tensor ，仅参数名不一致。  |
|  dim      |  axis  | 表示要堆叠的轴，仅参数名不一致。  |
|  out      | -      | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.stack([x1, x2, x3], out=y)

# Paddle 写法
paddle.assign(paddle.stack([x1, x2, x3]), y)
```
