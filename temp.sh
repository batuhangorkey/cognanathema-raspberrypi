# !/bin/bash

cpu=$(</sys/class/thermal/thermal_zone0/temp)
echo "$(date) @ $(hostname)"
echo "-------------------------------------------"
echo "CPU => $(vcgencmd measure_temp)"
echo "CPU => $((cpu/1000))'C"
echo "$(vcgencmd measure_temp | awk -F "[=']" '{print($2)}')"
