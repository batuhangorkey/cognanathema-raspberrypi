#!/bin/bash

while true
do
    cpu=$(</sys/class/thermal/thermal_zone0/temp)
    cpu_s=$(vcgencmd measure_temp | awk -F "[=']" '{print($2)}')
    cpu=$(bc <<< "scale=2; $cpu/1000")

    echo "$(date) @ $(hostname)"
    echo "-------------------------------------------"
    echo "CPU (thermal zone) => ${cpu}'C"
    echo "CPU (vcgencmd) => $cpu_s"
    
    sleep 1

    tput cuu 4
    # clear to the end of screen
    tput ed
done
