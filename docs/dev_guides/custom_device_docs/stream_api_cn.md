# Stream 接口

## create_stream 【optional】

### 接口定义

```c++
C_Status (*create_stream)(const C_Device device, C_Stream* stream)
```

### 接口说明

创建一个 stream 对象，stream 是框架内部用于执行异步任务的任务队列，同一 stream 中的任务按顺序执行。硬件不支持异步执行时该接口需要空实现。

### 参数

device - 使用的设备。

stream - 存储创建的 stream 对象。

## destroy_stream 【optional】

### 接口定义

```c++
C_Status (*destroy_stream)(const C_Device device, C_Stream stream)
```

### 接口说明

销毁一个 stream 对象。硬件不支持异步执行时该接口需要空实现。

### 参数

device - 使用的设备。

stream - 需要释放的 stream 对象。

## query_stream 【optional】

### 接口定义

```c++
C_Status (*query_stream)(const C_Device device, C_Stream stream)
```

### 接口说明

查询 stream 上的任务是否完成，如果没有实现，PaddlePaddle 会用 synchronize_stream 代替。

### 参数

device - 使用的设备。

stream - 需要查询的 stream。

## synchronize_stream 【optional】

### 接口定义

```c++
C_Status (*synchronize_stream)(const C_Device device, C_Stream stream)
```

### 接口说明

同步 stream，等待 stream 上所有任务完成。硬件不支持异步执行时该接口需要空实现。

### 参数

device - 使用的设备。

stream - 需要同步的 stream。

## stream_add_callback 【optional】

### 接口定义

```c++
C_Status (*stream_add_callback)(const C_Device device, C_Stream stream, C_Callback callback, void* user_data)
```

### 接口说明

添加一个主机回调函数到 stream 上。

### 参数

device - 使用的设备。

stream - 添加回调到该 stream 中。

callback - 回调函数。

user_data - 回调函数的参数。

## stream_wait_event 【optional】

### 接口定义

```c++
C_Status (*stream_wait_event)(const C_Device device, C_Stream stream, C_Event event)
```

### 接口说明

等待 stream 上的一个 event 完成。硬件不支持异步执行时该接口需要空实现。

### 参数

device - 使用的设备。

stream - 等待的 stream。

event - 等待的 event。
