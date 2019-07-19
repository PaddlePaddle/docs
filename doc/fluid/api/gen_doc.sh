#!/bin/bash

#for module in nn
#do
#  python gen_doc.py --module_name layers.${module} --module_prefix layers --output layers/${module} --to_multiple_files True
#done

for module in control_flow nn io ops tensor learning_rate_scheduler detection metric_op
do
  python gen_doc.py --module_name layers.${module} --module_prefix layers --output layers/${module}.rst
done 

for module in data_feeder dataset clip metrics executor initializer io nets optimizer profiler regularizer transpiler recordio_writer backward average profiler unique_name dygraph
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module}.rst
done

python gen_doc.py --module_name "" --module_prefix "" --output fluid.rst

#python gen_module_index.py layers.nn nn
python gen_module_index.py layers fluid.layers

python gen_index.py

