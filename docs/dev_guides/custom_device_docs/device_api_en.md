# Device APIs

## initialize 【optional】

### Definition

```c++
C_Status (*initialize)()
```

### Description

It initializes the device backend, such as the runtime or the driver. During the device registration, it is the first to be invoked. But if the API is not implemented, it will not be invoked.

## finalize 【optional】

### Definition

```c++
C_Status (*finalize)()
```

### Description

It deinitializes the device backend. For example, the deinitialization is performed during the exit of the runtime or the driver. The API is invoked till the end of the exit. But if it is not implemented, it will not be invoked.

## init_device 【optional】

### Definition

```c++
C_Status (*init_device)(const C_Device device)
```

### Description

It initializes the designated device and initializes all available devices during the plug-in registration. If not implemented, the API will not be invoked, and it is invoked only after initialization.

### Parameter

device - the device needed to be initialized。

## deinit_device 【optional】

### Definition

```c++
C_Status (*deinit_device)(const C_Device device)
```

### Description

It finalizes the designated device, and deallocate resources allocated to all devices. The API is inovked during the exit. If not implemented, it will not be inovked and it is invoked before finalization.

### Parameter

device - the device needed to be finalized

### Definition

## set_device 【required】

```c++
C_Status (*set_device)(const C_Device device)
```

### Description

It sets the current device, where following tasks are executed.

### Parameter

device - the device needed to be set

## get_device 【required】

### Definition

```c++
C_Status (*get_device)(const C_Device device)
```

### Description

It acquires the current device

### Parameter

device - to store the current device

## synchronize_device 【required】

### Definition

```c++
C_Status (*synchronize_device)(const C_Device device)
```

### Description

It synchronizes the device and waits for the completion of tasks on the device.

### Parameter

device - the device required to be synchronized

## get_device_count 【required】

### Definition

```c++
C_Status (*get_device_count)(size_t* count)
```

### Description

It counts available devices.

### Parameter

count - the number of available devices in storage

## get_device_list 【required】

### Definition

```c++
C_Status (*get_device_list)(size_t* devices)
```

### Description

It acquires the number list of all currently available devices.

### Parameter

devices - numbers of available devices in storage

## get_compute_capability 【required】

### Definition

```c++
C_Status (*get_compute_capability)(size_t* compute_capability)
```

### Description

It gets the computing capability of the device.

### Parameter

compute_capability - the computing capability of the stored device

## get_runtime_version 【required】

### Definition

```c++
C_Status (*get_runtime_version)(size_t* version)
```

### Description

It acquires the runtime version.

### Parameter

version - the runtime version in storage

## get_driver_version 【required】

### Definition

```c++
C_Status (*get_driver_version)(size_t* version)
```

### Description

It gets the driver version.

### Parameter

version - the version of the stored driver
