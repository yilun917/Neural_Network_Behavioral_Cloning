#!/bin/bash

pid=$(ps -e | grep keep_alive.sh | cut -c1-5 | tr -d "   ")
if [ -z "$pid" ]; then 
    echo "Script keep_alive.sh not running"
else 
    $(kill $pid) 
fi
