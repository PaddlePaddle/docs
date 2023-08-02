## [ 组合替代实现 ]torch.copysign

### [torch.copysign](https://pytorch.org/docs/stable/generated/torch.copysign.html#torch.copysign)

```python
torch.copysign(input,
          other,
          *,
          out=None)
```
创建一个新的浮点张量，其大小与` input `相同，正负符号与` other `相同

PaddlePaddle 目前无对应 API，可使用如下代码组合替代实现:

### 转写示例

#### out：指定输出
```python
# Pytorch 写法
torch.copysign(input, other, out=y)

# Paddle 写法
paddle.assign(paddle.abs(input) * paddle.sign(other), y)
```

```
