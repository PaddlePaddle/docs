## [组合替代实现] torch.cuda.comm.gather

### [torch.cuda.comm.gather](https://pytorch.org/docs/stable/generated/torch.cuda.comm.gather.html)
```python
torch.cuda.comm.gather(tensors, dim=0, destination=None, *, out=None)
```

将多个设备的张量集中起来，Paddle 无此 API，需要组合替代实现。

### 转写示例
```python
# PyTorch 写法
destination = 'cuda:0'
gathered_tensor = torch.cuda.comm.gather(tensors, destination=destination)

# Paddle 写法
def paddle_comm_gather(tensors, dim=0, destination=None, *, out=None):
    if destination is None:
        destination = paddle.CPUPlace()
    elif 'cuda' in destination:
        destination = paddle.CUDAPlace(int(destination.split(':')[-1]))

    gathered_tensors = [t.cuda(destination) if 'cuda' in t.place.__str__() else t.cpu() for t in tensors]

    gathered_tensor = paddle.concat(gathered_tensors, axis=dim)

    if out is not None:
        out.copy_(gathered_tensor)
        return out

    return gathered_tensor

destination = 'gpu:0'
gathered_tensor = paddle_comm_gather(tensors, dim=dim, destination=destination)
```
