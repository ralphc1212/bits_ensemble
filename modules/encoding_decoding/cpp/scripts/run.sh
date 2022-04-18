#!/bin/bash

set -e # fail and exit on any command erroring

root_dir=$(cd `dirname $0`/..; pwd)
build_dir=${root_dir}/build
exec=${build_dir}/bit_ensemble
workload_dir=${root_dir}/inputs/
encoded_dir=${root_dir}/encoded/
tmp_dir=${root_dir}/scripts/tmp/
buffer_size_list=(2048)
num_block_data_list=(256)

# Configurations
nn_list=('vgg11' 'vgg11-8')

start_time=`echo $(($(date +%s%N)/1000000))`
for nn in ${nn_list[*]}
do
  output_path=${tmp_dir}${nn}.out
  echo '' > ${output_path}
  for buffer_size in ${buffer_size_list[*]}
  do
    for num_block_data in ${num_block_data_list[*]}
    do
      for file in ${workload_dir}${nn}.*.in
      do
        layer_name=`basename ${file}`
        encoded_path=${encoded_dir}'encoded_'${buffer_size}'_'${num_block_data}'_'${layer_name[@]/.in/.bin}
        echo 'Layer ['${layer_name}'] Buffer Size ['${buffer_size}'] # Block Data ['${num_block_data}']'
        echo -n ${layer_name}' '${buffer_size}' '${num_block_data}' ' >> ${output_path}
        echo `${exec} ${buffer_size} ${num_block_data} ${file} ${encoded_path}` >> ${output_path}
      done
    done
  done
done
end_time=`echo $(($(date +%s%N)/1000000))`
time_cost=$((${end_time} - ${start_time}))
echo 'Time Cost '$((${time_cost}/1000))'(s)'

for nn in ${nn_list[*]}
do
  python3 print_result.py ${nn} ${encoded_dir} ${tmp_dir}
  break
done
