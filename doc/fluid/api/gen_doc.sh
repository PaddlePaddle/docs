#!/bin/bash

mkdir -p layers

for module in control_flow io nn ops tensor learning_rate_scheduler detection metric_op
do
  python gen_doc.py --module layers.${module} --module_prefix layers > layers/${module}.rst
done 

for module in data_feeder dataset clip metrics executor initializer io nets optimizer profiler regularizer transpiler recordio_writer backward average profiler unique_name
do
  python gen_doc.py --module ${module} --module_prefix ${module} > ${module}.rst
done

python gen_doc.py --module "" --module_prefix "" > fluid.rst

python gen_index.py
