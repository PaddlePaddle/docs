## torch.stack
### [torch.stack](https://pytorch.org/docs/1.13/generated/torch.stack.html?highlight=stack#torch.stack)

```python
torch.stack(tensors,
            dim=0,
            *,
            out=None)
```

### [paddle.stack](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/stack_cn.html#stack)

```python
paddle.stack(x,
             axis=0,
             name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensors       | x            | 输入的 Tensor。                                      |
| dim           | axis         | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.stack([x1, x2, x3], out=y)

# Paddle 写法
y = paddle.stack([x1, x2, x3])
```
