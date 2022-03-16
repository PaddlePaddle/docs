# Event接口

## create_event 【required】

### 接口定义

```c++
C_Status (*create_event)(const C_Device device, C_Event* event)
```

### 接口说明

创建一个event对象。

### 参数

device - 使用的设备。

event - 存储创建的event对象。

## destroy_event 【required】

### 接口定义

```c++
C_Status (*destroy_event)(const C_Device device, C_Event event)
```

### 接口说明

销毁一个event对象。

### 参数

device - 使用的设备。

event - 需要释放的event对象。

## record_event 【required】

### 接口定义

```c++
C_Status (*record_event)(const C_Device device, C_Stream stream, C_Event event)
```

### 接口说明

在stream上记录event。

### 参数

device - 使用的设备。

stream - 在该stream上记录event。

event - 被记录的event。

## query_event 【optional】

### 接口定义

```c++
C_Status (*query_event)(const C_Device device, C_Event event)
```

### 接口说明

查询event是否完成，如果没有实现，PaddlePaddle 会用 synchronize_event 代替

### 参数

device - 使用的设备。

event - 需要查询的event对象。

## synchronize_event 【required】

### 接口定义

```c++
C_Status (*synchronize_event)(const C_Device device, C_Event event)
```

### 接口说明

同步event，等待event完成。

### 参数

device - 使用的设备。

event - 需要同步的event。
