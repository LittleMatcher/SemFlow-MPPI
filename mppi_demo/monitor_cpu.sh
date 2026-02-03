#!/bin/bash
# CPU监控脚本 - 显示Python进程的CPU使用情况

echo "监控Python进程的CPU使用情况（按Ctrl+C停止）"
echo "================================================"

while true; do
    clear
    date
    echo ""
    echo "Python进程CPU使用情况："
    echo "PID      CPU%    MEM%    命令"
    echo "-------- ------- ------- --------------------------------"
    ps aux | grep python | grep -v grep | grep -v monitor | awk '{printf "%-8s %-7s %-7s %s\n", $2, $3, $4, $11}'
    echo ""
    echo "总体CPU使用："
    mpstat 1 1 | tail -n 1
    echo ""
    echo "活跃的CPU核心数："
    ps -eLo psr | tail -n +2 | sort -u | wc -l
    sleep 2
done
