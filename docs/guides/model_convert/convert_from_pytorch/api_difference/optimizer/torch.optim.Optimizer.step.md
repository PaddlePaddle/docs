## [ torch 参数更多 ]torch.optim.Optimizer.step

### [torch.optim.Optimizer.step](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html#torch-optim-optimizer-step)

```python
torch.optim.Optimizer.step(closure)
```

### [paddle.optimizer.Optimizer.step](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Optimizer_cn.html#step)

```python
paddle.optimizer.Optimizer.step()
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

|  PyTorch   | PaddlePaddle |        备注        |
|  --------  |  ----------  |  ----------------  |
| closure |  -  | 重新评估模型并返回损失的闭包, Paddle 无此参数，暂无转写方式。 |


### 转写示例
####
```python
# Pytorch 写法
torch.optim.Optimizer.step()

# Paddle 写法
paddle.optimizer.Optimizer.step()
```
