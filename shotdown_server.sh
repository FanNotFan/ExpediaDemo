#!/bin/bash
#获取5000端口的第七列的值给perpid变量
arr=(5000 5005)
# shellcheck disable=SC2068
for var in ${arr[@]};
do
 perpid=`lsof -Pnl +M -i4 | grep $var | grep python | awk '{print $2}'`
 if [[ $perpid ]];then
         #从后开始删除变量，删除到第一个/停止，赋值给apid
         apid=${perpid%/*}
         echo "Get the process number of port $var:$apid"
         #杀死进程
         kill -9 $apid
         echo "Kill the process occupying port $var(pid:$apid)"
 else
         echo "No process occupies port $var"
 fi
done

if [ $? -ne 0 ]; then
    echo "shutdown failed"
else
    echo "shutdown succeed"
fi
exec /bin/bash