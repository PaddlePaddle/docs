## [参数不一致]torch.floor
### [torch.floor](https://pytorch.org/docs/1.13/generated/torch.floor.html?highlight=torch+floor#torch.floor)

```python
torch.floor(input,
            *,
            out=None)
```

### [paddle.floor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/floor_cn.html#floor)

```python
paddle.floor(x,
             name=None)
```

其中 Pytorch 和 Paddle 的 `input` 参数用法不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的 Tensor ，torch 支持整型和浮点型，paddle 仅支持浮点型。当输入为整型时，需要进行转写。  |
|  out  | - |  表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |

### 转写示例
#### input：输入为整型
```python
# Pytorch 写法
torch.floor(torch.tensor([2, 3, 8, 7]))

# Paddle 写法
paddle.floor(paddle.to_tensor([2, 3, 8, 7]).astype('float32'))
```
#### out：指定输出
```python
# Pytorch 写法
torch.floor(torch.tensor([-0.4, -0.2, 0.1, 0.3]), out=y)

# Paddle 写法
paddle.assign(paddle.floor(paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])), y)
```
