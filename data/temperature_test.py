import psutil
import time
from datetime import datetime

f = open("monitoring_temp.txt", "w+")
print("Monitoring temperature started")
i = 0
while True:
    try:
        f.write("Monitoring: " + str(i) + " : " + str(datetime.now()) + "\n")
        f.write(str(psutil.sensors_temperatures()) + "\n")
        time.sleep(1)
        i += 1
    except KeyboardInterrupt:
        print("Monitoring temperature ended")
        f.close()