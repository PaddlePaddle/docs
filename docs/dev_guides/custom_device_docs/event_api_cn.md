# Event 接口

## create_event 【required】

### 接口定义

```c++
C_Status (*create_event)(const C_Device device, C_Event* event)
```

### 接口说明

创建一个 event 对象，event 被框架内部用于同步不同 stream 之间的任务。硬件不支持异步执行时该接口需要空实现。

### 参数

device - 使用的设备。

event - 存储创建的 event 对象。

## destroy_event 【required】

### 接口定义

```c++
C_Status (*destroy_event)(const C_Device device, C_Event event)
```

### 接口说明

销毁一个 event 对象。硬件不支持异步执行时该接口需要空实现。

### 参数

device - 使用的设备。

event - 需要释放的 event 对象。

## record_event 【required】

### 接口定义

```c++
C_Status (*record_event)(const C_Device device, C_Stream stream, C_Event event)
```

### 接口说明

在 stream 上记录 event。硬件不支持异步执行时该接口需要空实现。

### 参数

device - 使用的设备。

stream - 在该 stream 上记录 event。

event - 被记录的 event。

## query_event 【optional】

### 接口定义

```c++
C_Status (*query_event)(const C_Device device, C_Event event)
```

### 接口说明

查询 event 是否完成，如果没有实现，PaddlePaddle 会用 synchronize_event 代替。

### 参数

device - 使用的设备。

event - 需要查询的 event 对象。

## synchronize_event 【required】

### 接口定义

```c++
C_Status (*synchronize_event)(const C_Device device, C_Event event)
```

### 接口说明

同步 event，等待 event 完成。硬件不支持异步执行时该接口需要空实现。

### 参数

device - 使用的设备。

event - 需要同步的 event。
