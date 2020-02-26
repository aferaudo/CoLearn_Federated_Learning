#!/bin/sh


i=0

while [ $i -lt $1 ]
do
	cp /rom/etc/config/firewall /etc/config/firewall
	/etc/init.d/firewall restart	
	echo "firewall restarted"
	echo "starting osmud"
	./startup.sh &
	sleep $2 
	echo "killing osmud.."
	/etc/init.d/osmud stop
	echo "killed"	
	i=`expr $i + 1`
	
	cp /var/log/osmud_perf.log result/test_$i.txt 
done
