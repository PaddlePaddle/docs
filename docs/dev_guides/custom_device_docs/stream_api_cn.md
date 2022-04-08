# Stream接口

## create_stream 【required】

### 接口定义

```c++
C_Status (*create_stream)(const C_Device device, C_Stream* stream)
```

### 接口说明

创建一个stream对象，stream是框架内部用于执行异步任务的任务队列，同一stream中的任务按顺序执行。硬件不支持异步执行时该接口需要空实现。

### 参数

device - 使用的设备。

stream - 存储创建的stream对象。

## destroy_stream 【required】

### 接口定义

```c++
C_Status (*destroy_stream)(const C_Device device, C_Stream stream)
```

### 接口说明

销毁一个stream对象。硬件不支持异步执行时该接口需要空实现。

### 参数

device - 使用的设备。

stream - 需要释放的stream对象。

## query_stream 【optional】

### 接口定义

```c++
C_Status (*query_stream)(const C_Device device, C_Stream stream)
```

### 接口说明

查询stream上的任务是否完成，如果没有实现，PaddlePaddle 会用 synchronize_stream 代替。

### 参数

device - 使用的设备。

stream - 需要查询的stream。

## synchronize_stream 【required】

### 接口定义

```c++
C_Status (*synchronize_stream)(const C_Device device, C_Stream stream)
```

### 接口说明

同步stream，等待stream上所有任务完成。硬件不支持异步执行时该接口需要空实现。

### 参数

device - 使用的设备。

stream - 需要同步的stream。

## stream_add_callback 【optional】

### 接口定义

```c++
C_Status (*stream_add_callback)(const C_Device device, C_Stream stream, C_Callback callback, void* user_data)
```

### 接口说明

添加一个主机回调函数到stream上。

### 参数

device - 使用的设备。

stream - 添加回调到该stream中。

callback - 回调函数。

user_data - 回调函数的参数。

## stream_wait_event 【required】

### 接口定义

```c++
C_Status (*stream_wait_event)(const C_Device device, C_Stream stream, C_Event event)
```

### 接口说明

等待stream上的一个event完成。硬件不支持异步执行时该接口需要空实现。

### 参数

device - 使用的设备。

stream - 等待的stream。

event - 等待的event。
