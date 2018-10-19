# How to use timeline tool to do profile

## <span id="local">Local</span>

1. Add `profiler.start_profiler(...)`和`profiler.stop_profiler(...)` to the main training loop. After run, the code will generate a profile record file `/tmp/profile`. **Warning**: Please do not run too many batches when use profiler to record timeline information, for the profile record will grow with the batch number.

	```python
    for pass_id in range(pass_num):
        for batch_id, data in enumerate(train_reader()):
            if pass_id == 0 and batch_id == 5:
                profiler.start_profiler("All")
            elif pass_id == 0 and batch_id == 10:
                profiler.stop_profiler("total", "/tmp/profile")
            exe.run(fluid.default_main_program(),
                    feed=feeder.feed(data),
                    fetch_list=[])
	            ...
	```

1. Run `python paddle/tools/timeline.py` to process `/tmp/profile`, it will generate another
file `/tmp/timeline` by default. You can change the path by cmd parameter, please take a look at
[timeline.py](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/timeline.py) for details.
```python
python Paddle/tools/timeline.py --profile_path=/tmp/profile --timeline_path=timeline
```

1. Open chrome and visit <chrome://tracing/>, use `load` button to load the generated `timeline` file.

	![chrome tracing](./tracing.jpeg)

1. The resulting timeline should be like:


	![chrome timeline](./timeline.jpeg)
	
## Distributed
This tool can support distributed train programs(pserver and trainer) too.

1. Open traniner profiler just like how to use in [local](#local).

1. Open pserver profiler: add some enviroment variables, eg:
```
FLAGS_rpc_server_profile_period=10 FLAGS_rpc_server_profile_path=./tmp/pserver python train.py
```

1. Merge pservers' and trainers' profiler file, eg:
```
python /paddle/tools/timeline.py
    --profile_path trainer0=local_profile_10_pass0_0,trainer1=local_profile_10_pass0_1,pserver0=./pserver_0,pserver1=./pserver_1
    --timeline_path ./dist.timeline
```
 
1. Load `dist.timeline` in chrome://tracing
