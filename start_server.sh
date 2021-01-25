#!/bin/bash
#获取5000端口的第七列的值给perpid变量
arr=(5000 8080)
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

source activate python3.7.3_dev

# for ((i = 0; i < ${#services[@]}; i++))
# do
#     eval "${services[$i]}"
# done

# cd /Users/xyao/Library/Mobile\ Documents/com~apple~CloudDocs/JupyterHome/Simplification/ExpediaDemo
cd /Users/hiCore/Develop/Workspace_Work/Workspace_Pycharm/Expedia_WorkSpace/ExpediaDemo
git pull

flask run & 
python tools/websocket.py &

sleep 3
if [ $? -ne 0 ]; then
    echo "failed"
else
    echo "succeed"
    open /Applications/Google\ Chrome.app/ http://127.0.0.1:5000
fi
exec /bin/bash