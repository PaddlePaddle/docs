# Device 接口

## initialize 【optional】

### 接口定义

```c++
C_Status (*initialize)()
```

### 接口说明

初始化硬件后端，例如初始化硬件 Runtime 或者 Driver 。在注册硬件时，最先被调用，不实现该接口则不调用。

## finalize 【optional】

### 接口定义

```c++
C_Status (*finalize)()
```

### 接口说明

去初始化硬件后端，例如硬件 Runtime 或者 Driver 退出时去初始化。在退出时最后被调用，不实现该接口则不调用。

## init_device 【optional】

### 接口定义

```c++
C_Status (*init_device)(const C_Device device)
```

### 接口说明

初始化指定硬件设备，会在插件注册时对所有可用设备进行初始化，不实现该接口则不调用。该接口在 initialize 之后被调用。

### 参数

device - 需要初始化的设备。

## deinit_device 【optional】

### 接口定义

```c++
C_Status (*deinit_device)(const C_Device device)
```

### 接口说明

去初始化指定硬件设备，释放所有该设备分配的资源，在退出时调用，不实现该接口则不调用。该接口在 finalize 之前被调用。

### 参数

device - 需要去初始化的设备。

### 接口定义

## set_device 【required】

```c++
C_Status (*set_device)(const C_Device device)
```

### 接口说明

设置当前使用的硬件设备，后续的任务执行在该设备上。

### 参数

device - 需要设置的设备。

## get_device 【required】

### 接口定义

```c++
C_Status (*get_device)(const C_Device device)
```

### 接口说明

获取当前使用的硬件设备。

### 参数

device - 存储当前使用的设备。

## synchronize_device 【required】

### 接口定义

```c++
C_Status (*synchronize_device)(const C_Device device)
```

### 接口说明

同步设备，等待指定设备上所有任务完成。

### 参数

device - 需要同步的设备。

## get_device_count 【required】

### 接口定义

```c++
C_Status (*get_device_count)(size_t* count)
```

### 接口说明

查询可用设备数量。

### 参数

count - 存储可用设备数量。

## get_device_list 【required】

### 接口定义

```c++
C_Status (*get_device_list)(size_t* devices)
```

### 接口说明

获取当前可用所有设备的设备号列表。

### 参数

devices - 存储可用设备号。

## get_compute_capability 【required】

### 接口定义

```c++
C_Status (*get_compute_capability)(size_t* compute_capability)
```

### 接口说明

获取设备算力。

### 参数

compute_capability - 存储设备算力。

## get_runtime_version 【required】

### 接口定义

```c++
C_Status (*get_runtime_version)(size_t* version)
```

### 接口说明

获取运行时版本号。

### 参数

version - 存储运行时版本号。

## get_driver_version 【required】

### 接口定义

```c++
C_Status (*get_driver_version)(size_t* version)
```

### 接口说明

获取驱动版本号。

### 参数

version - 存储驱动版本号。
