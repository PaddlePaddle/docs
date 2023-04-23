## [torch 参数更多 ]torch.nonzero
### [torch.nonzero](https://pytorch.org/docs/1.13/generated/torch.nonzero.html#torch.nonzero)

```python
torch.nonzero(input,
              *,
              out=None,
              as_tuple=False)
```

### [paddle.nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nonzero_cn.html)

```python
paddle.nonzero(x,
               as_tuple=False)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input | x | 表示输入的 Tensor ，仅参数名不一致。  |
|  out  | - | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |
| as_tuple | as_tuple | bool 类型表示输出数据的格式，默认 False 时，输出一个张量，True 时输出一组一维张量。  |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.nonzero(x, out=y)

# Paddle 写法
paddle.assign(paddle.nonzero(x), y)
```
