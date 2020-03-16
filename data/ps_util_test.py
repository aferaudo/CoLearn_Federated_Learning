import argparse

import psutil
from psutil._common import bytes2human
import time
import logging
import threading
import socket
from datetime import datetime


af_map = {
    socket.AF_INET: 'IPv4',
    socket.AF_INET6: 'IPv6',
    psutil.AF_LINK: 'MAC',
}

duplex_map = {
    psutil.NIC_DUPLEX_FULL: "full",
    psutil.NIC_DUPLEX_HALF: "half",
    psutil.NIC_DUPLEX_UNKNOWN: "?",
}


parser = argparse.ArgumentParser(description="Process monitoring")
parser.add_argument(
    "--pid", "-p", type=int, default=None, help="process pid to monitor"
)
parser.add_argument(
    "--network", "-n", type=str, default=None, help="Start monitor a network interface"
)



def monitor_cpu(pid):
    f = open("monitoring_cpu.txt", "w+")
    p = psutil.Process(pid)
    logging.info(p)
    f.write(str(p))
    f.write("\n")
    for i in range(3000):
        f.write("Monitoring: " + str(i) + " " + str(datetime.now()) + "\n" )
        f.write(str(p.cpu_times()) + "\n")
        # f.write("Num of threads: " + str(p.thread()))
        f.write(str(p.cpu_percent(interval=1.0)) + "\n")
    
    f.close()

def monitor_network(interface):
    f = open("monitoring_network.txt", "w+")
    
    for i in range (3000):
        stats = psutil.net_if_stats()
        io_counters = psutil.net_io_counters(pernic=True)
        if not interface in stats or not interface in io_counters:
            logging.info("Interface not valid")
            break

        f.write("Monitoring: " + str(i) + " " + str(datetime.now()) + "\n" )
        # print("    stats          : ", end='')
        # print("speed=%sMB, duplex=%s, mtu=%s, up=%s" % (
        #     stats[interface].speed, duplex_map[stats[interface].duplex], stats[interface].mtu,
        #     "yes" if stats[interface].isup else "no"))
        io = io_counters[interface]
        if i == 0:
            # In this way the stats start alway from zero
            initial_rcv = io.bytes_recv
            initial_pkts_rcv = io.packets_sent
            initial_errin = io.errin
            initial_dropin = io.dropin
            initial_sent = io.bytes_sent
            initial_pkts_sent = io.packets_sent
            initial_errout = io.errout
            initial_dropout = io.dropout

        f.write("    incoming       : ")
        f.write("bytes=%s, pkts=%s, errs=%s, drops=%s" % (
            bytes2human((io.bytes_recv - initial_rcv)), str(io.packets_recv - initial_pkts_rcv), str(io.errin - initial_errin),
            str(io.dropin - initial_dropin)))
        f.write("\n")
        f.write("    outgoing       : ")
        f.write("bytes=%s, pkts=%s, errs=%s, drops=%s" % (
            bytes2human((io.bytes_sent - initial_sent)), str(io.packets_sent - initial_pkts_sent), str(io.errout - initial_errout),
            str(io.dropout-initial_dropout)))
        f.write("\n\n")
        time.sleep(1)
    f.close()


def main(argv):

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    threads = list()

    # CPU thread starting
    if args.pid != None:
        t1 = threading.Thread(target=monitor_cpu, args=(args.pid, ))
        logging.info("Starting CPU monitor thread")
        t1.start()
        threads.append(t1)
    
    # Network thread starting
    if args.network != None:
        logging.info("Starting network monitoring")
        t2 = threading.Thread(target=monitor_network, args=(args.network, ))
        t2.start()
        threads.append(t2)

    logging.info("Waiting for the threads ending")
    for i in range(len(threads)):
        threads[i].join()
    
    logging.info("Join complete")

if __name__ == "__main__":

    args = parser.parse_args()
    main(args)