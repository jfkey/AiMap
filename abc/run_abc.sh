#!/bin/bash

# run abc for each file in list
# bash run_abc.sh <path_abc> <path_benchmark> <timeout> <path_lib>
# eg: bash run_abc.sh ../bin/abc ../benchmark/EPFL 10m ../asap7.lib
#bash run_abc.sh  cmake-build-debug/abc /home/liujunfeng/benchmarks/random-arith 10m /home/liujunfeng/SLAP/ML-Mapper/abc/asap7_clean.lib
# bash run_abc.sh  cmake-build-debug/abc /home/liujunfeng/benchmarks/random_control 10m /home/liujunfeng/SLAP/ML-Mapper/abc/asap7_clean.lib
#bash run_abc.sh  cmake-build-debug/abc /home/liujunfeng/benchmarks/comb-exp 10m /home/liujunfeng/SLAP/ML-Mapper/abc/asap7_clean.lib
#bash run_abc.sh  cmake-build-debug/abc /home/liujunfeng/benchmarks/control-exp 10m /home/liujunfeng/SLAP/ML-Mapper/abc/asap7_clean.lib

binary=$(echo "$1" | awk -F "/" '{print $NF}')
timestamp=$(date +%Y%m%d%H%M%S)
csv="${timestamp}_$binary.csv"
#log="${timestamp}_$binary.log"
touch "$csv"
#touch "$log"
echo "name, command, input, output, lat, gates, edge, area, delay, lev, stime_gates, stime_gates%, stime_cap(ff), stime_cap%, stime_Area, stime_Area%, stime_Delay(ps), stime_Delay%" >> $csv

files=$(find "$2" -name "*.")

for element in ${files[@]}
do
    echo "process $element"
    command="read_lib $4 ;read $element; strash; map -v -r; topo; print_stats; stime";
    outputs=$(timeout $3 $1 -c "$command";)
#    echo $outputs >> $log

    numbers=($(echo $outputs | grep -Eo '[0-9]+(\.[0-9]+)?'))
    size=${#numbers[@]}
    name=$(echo "$element" | awk -F "/" '{print $NF}')

    ret="$name, $command, ${numbers[$size-28]}, ${numbers[$size-27]}, ${numbers[$size-27]},  ${numbers[$size-25]}, ${numbers[$size-24]}, ${numbers[$size-23]}, ${numbers[$size-22]}, ${numbers[$size-21]}, ${numbers[$size-18]}, ${numbers[$size-16]}, ${numbers[$size-13]}, ${numbers[$size-11]}, ${numbers[$size-8]}, ${numbers[$size-6]}, ${numbers[$size-3]}, ${numbers[$size-1]}"
    echo $ret >> $csv

#    command="stime -p";
#    outputs=$(timeout $3 $1 -c "$command";)
#    logele="$element.log"
#    touch "$logele"
#    echo $outputs >> $logele
done

