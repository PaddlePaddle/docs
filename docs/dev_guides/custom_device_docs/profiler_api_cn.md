# Profiler 接口

## profiler_initialize 【optional】

### 接口定义

```c++
C_Status (*profiler_initialize)(C_Profiler prof, void** user_data)
```

### 接口说明

初始化硬件 Profiler

### 参数

prof - C_Profiler 对象。

user_data - 用户数据。

## profiler_finalize 【optional】

### 接口定义

```c++
C_Status (*profiler_finalize)(C_Profiler prof, void* user_data)
```

### 接口说明

去初始化硬件 Profiler

### 参数

prof - C_Profiler 对象。

user_data - 用户数据。

## profiler_prepare_tracing 【optional】

### 接口定义

```c++
C_Status (*profiler_prepare_tracing)(C_Profiler prof, void* user_data)
```

### 接口说明

Profiler 准备收集数据

### 参数

prof - C_Profiler 对象。

user_data - 用户数据。

## profiler_start_tracing 【optional】

### 接口定义

```c++
C_Status (*profiler_start_tracing)(C_Profiler prof, void* user_data)
```

### 接口说明

Profiler 开始收集数据

### 参数

prof - C_Profiler 对象。

user_data - 用户数据。

## profiler_stop_tracing 【optional】

### 接口定义

```c++
C_Status (*profiler_stop_tracing)(C_Profiler prof, void* user_data)
```

### 接口说明

Profiler 停止收集数据

### 参数

prof - C_Profiler 对象。

user_data - 用户数据。

## profiler_collect_trace_data 【optional】

### 接口定义

```c++
C_Status (*profiler_collect_trace_data)(C_Profiler prof, uint64_t start_ns, void* user_data)
```

### 接口说明

Profiler 数据转换，调用 `profiler_add_runtime_trace_event` ， `profiler_add_device_trace_event` 转换为 Paddle Profiler 使用的数据。

### 参数

prof - C_Profiler 对象。

start_ns - 时间戳。

user_data - 用户数据。
