## [ torch 参数更多 ]torch.mv
### [torch.mv](https://pytorch.org/docs/1.13/generated/torch.mv.html?highlight=torch+mv#torch.mv)
```python
torch.mv(input, vec, out=None)
```

### [paddle.mv](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/mv_cn.html)

```python
paddle.mv(x, vec, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                   |
| vec         | vec           | 表示输入的 Tensor 。                   |
| out         | -            | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写 。                   |

### 转写示例

#### out：指定输出
```python
# Pytorch 写法
torch.mv(x, vec, out=y)

# Paddle 写法
paddle.assign(paddle.mv(x, vec), y)
```
