#!/bin/bash

for module in layers dataset clip metrics executor initializer io nets optimizer profiler regularizer transpiler backward profiler unique_name dygraph framework
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name fluid --to_multiple_files True
  python gen_module_index.py ${module}  fluid.${module}
done

python gen_doc.py --module_name "" --module_prefix "" --output fluid --output_name fluid --to_multiple_files True
python gen_module_index.py fluid  fluid

for module in fleet
do
    python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name distributed --to_multiple_files True
    python gen_module_index.py ${module} distributed.${module}
done

python gen_doc.py --module_name "" --module_prefix "" --output distributed --output_name distributed --to_multiple_files True
python gen_module_index.py distributed distributed

for module in meta_optimizers metrics dataset runtime utils
do
    python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name fleet --to_multiple_files True
    python gen_module_index.py ${module} fleet.${module}
done

python gen_doc.py --module_name "" --module_prefix "" --output fleet --output_name fleet --to_multiple_files True
python gen_module_index.py fleet fleet

# tensor
for module in math random stat linalg search
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name tensor --to_multiple_files True --output_dir tensor
  python gen_module_index.py tensor.${module} ${module}
done

python gen_module_index.py tensor paddle.tensor

for module in math manipulation linalg
do
  python gen_doc.py --module_name tensor.${module} --module_prefix tensor.${module} --output tensor/${module} --output_name complex --to_multiple_files True --output_dir complex
  python gen_module_index.py complex.tensor.${module} ${module}
done

python gen_module_index.py complex.tensor tensor
python gen_module_index.py complex paddle.complex
python gen_module_index.py framework paddle.framework


# nn
for module in loss
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name nn --to_multiple_files True --output_dir nn
  python gen_module_index.py nn.${module} ${module}
done

python gen_doc.py --module_name "" --module_prefix "" --output nn --output_name nn --to_multiple_files True
python gen_module_index.py nn paddle.nn

# index.rst
python gen_index.py

