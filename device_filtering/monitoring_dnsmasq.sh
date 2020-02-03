#!/bin/sh

# THIS SCRIPT MUST BE EXECUTED ON THE ROUTER RUNNING OSMUD!

# GENERATE THE KEYS ON THE ROUTER
# If you are executing on openwrt this is the correct procedure
# Generation private key
# dropbearkey -t rsa -f ~/.ssh/id_rsa
# 
# Generation Public key for openssh
# dropbearkey -y -f ~/.ssh/id_rsa | grep "^ssh-rsa " > ~/.ssh/id_rsa.pub
# 
# Copy the pub key on the remote_host
# cat ~/.ssh/id_rsa.pub | ssh <user>@<ip_address> "cat >> ~/.ssh/authorized_keys"
#
# Connection without password: you need to specify the key location
# ssh <user>@<ip_address> -i ~/.ssh/id_rsa

# This script is created to coordinate osmud with the federated_coordinator in order to filter the devices
# that can participate to the learning phase
# Must be executed on a router with openwrt

if [ $# -eq 0 ]
then
	echo "Missing options! Before to use this script you must read the -h option"
        echo "(run $0 -h for help)"
        echo ""
        exit 0
fi
REMOTE_USER=""
REMOTE_PATH=""
INTERVAL_OF_SCANNING=40
while getopts "hu:p:i:" OPTION; do
        case $OPTION in
		
		u)
                        REMOTE_USER=$OPTARG
                        ;;
                p)
       			REMOTE_PATH=$OPTARG
       			;;         	
                i)
                        INTERVAL_OF_SCANNING=$OPTARG
                        ;;
                h)
                        echo "The remote script executed is a python script, called file_upgrader.py
Pay attention: The script works only with osmud and needs a secure connection without password between the device that call the script (router as openssh client) and the device coordinator. Furthermore the coordinator's ip address must be registred as www.mfs.example.com in the file /etc/hosts. To launch this script in background you can use $0 -p <script_path> -u <user> -i <interval> >/dev/null &"
                        echo "Usage:"
                        echo "$0 -h "
                        echo "$0 -p <script_path> -u <user> -i <interval>"
                        echo ""
                        echo "	 -u	coordinator user"
                     	echo "	 -p	script's path on the coordinator device (this is not so secure for now, because an attacker can execute another script on the other side)"
                        echo "	 -i	interval of scanning the file /var/log/dnsmasq.txt (default 40 seconds)"
                        echo "   -h     help"
                        exit 0
                        ;;

        esac
done

while true;do
	input="/var/log/dhcpmasq.txt"
	while read -r line
	do
		echo $line
		COMMAND=$(echo $line | awk -F "|" '{ print $2}')
		MUD_URL=$(echo $line | awk -F "|" '{ print $7}')
		IP_TO_VALIDATE=$(echo $line | awk -F "|" '{ print $10}')
		# echo $COMMAND
		# echo $MUD_URL
		# echo $IP_TO_VALIDATE
		if [ "$MUD_URL" = "-" -o "$COMMAND" = "OLD" ]; then
			echo "not valid"
		else	
			ssh $REMOTE_USER@www.mfs.example.com "cd $REMOTE_PATH/; python file_upgrader.py -c $COMMAND -i $IP_TO_VALIDATE" < /dev/null
		fi
		sleep 1
	done < "$input"
	sleep $INTERVAL_OF_SCANNING
done
