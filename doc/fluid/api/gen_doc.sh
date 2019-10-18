#!/bin/bash

#for module in nn
#do
#  python gen_doc.py --module_name layers.${module} --module_prefix layers --output layers/${module} --to_multiple_files True
#done

#for module in control_flow nn io ops tensor learning_rate_scheduler detection metric_op
#do
#  python gen_doc.py --module_name layers.${module} --module_prefix layers --output layers/${module}.rst
#done 

for module in layers data_feeder dataset clip metrics executor initializer io nets optimizer profiler regularizer transpiler backward average profiler unique_name dygraph
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --to_multiple_files True
  python gen_module_index.py ${module}  fluid.${module}
done

python gen_doc.py --module_name "" --module_prefix "" --output fluid --to_multiple_files True
python gen_module_index.py fluid  fluid

python gen_index.py

