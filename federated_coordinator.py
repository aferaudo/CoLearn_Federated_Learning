#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2013 Roger Light <roger@atchoo.org>
#
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Eclipse Distribution License v1.0
# which accompanies this distribution.
#
# The Eclipse Distribution License is available at
#   http://www.eclipse.org/org/documents/edl-v10.php.
#
# Contributors:
#    Roger Light - initial implementation

# This example shows how you can use the MQTT client in a class.

# BROKER SETTINGS (First stop the default service launchctl stop homebrew.mxcl.mosquitto)
# /usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf --> launch mosquitto
# mosquitto_pub -t topic/state -m "(192.168.1.7, TRAINING)" --> event to publish

# Important things to understand:
# If two workers have the same id at the end only one will be created

import sys
import getopt
import torch
import syft as sy
from torch import optim
import time
from syft.frameworks.torch.federated import utils

import paho.mqtt.client as mqtt
from torchvision import datasets, transforms # datasets is used only to do some tests
from event_parser import EventParser
import client_federated as cf
from threading import Timer


class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 1
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False

class Coordinator(mqtt.Client):

    def __init__(self, window):
        super(Coordinator, self).__init__()
        self.training_lower_bound = 2
        self.training_upper_bound = 100
        self.event_served = 0
        self.__hook = sy.TorchHook(torch)
        self.__hook.local_worker.is_client_worker = False # Set the local worker as a server: All the other worker will be registered in __known_workers
        self.server = self.__hook.local_worker
        self.window = 20.0
        # TODO load the model from a file if it is present, otherwise create a new one
        self.model = cf.Net()
        self.args = Arguments()
        


    def on_connect(self, mqttc, obj, flags, rc):
        print("Connected!")

    def on_message(self, mqttc, obj, msg):
        print(msg.topic+" "+str(msg.qos)+" "+str(msg.payload))
        parser = EventParser(msg.payload)
        ip_address = parser.ip_address()
        state = parser.state()
        if ip_address != -1 and state != None:
            print(ip_address + " " + state )
            
            if state == "TRAINING":
                self.event_served += 1
                print("Beharvior training")
                
                worker = sy.VirtualWorker(self.__hook, ip_address) # registration
               
                print(self.server._known_workers)
                
                if self.event_served == 1: # Start the timer after received an event
                    print("Timer starting")
                    t = Timer(self.window, self.__starting_training)
                    t.start()
                    

            elif state == "INFERENCE":
                # TODO inference
                print("Behavior inference")
            
            elif state == "NOT_READY":
                self.__remove_safely_known_workers(key=ip_address)
                print(self.server._known_workers)

            else:
                print("No behavior defined for this event")

            # TODO When a client is no more available we have to remove it from the workers list (training, inference or both?)


        else:
            # TODO What we have to do if the ip or the state is not valid
            print("not ok!")

    def on_publish(self, mqttc, obj, mid):
        print("mid: "+str(mid))

 

    def run(self, host, port, topic, keepalive):
        self.connect(host, port)
        self.subscribe(topic, 0)

        rc = 0
        while rc == 0:
            rc = self.loop_forever()
        return rc
    
    def __starting_training(self):
        # TODO Now in both the case, (1)when we have an adequate number of client and (2) when not, 
        # The window will be recreated at the next event (This beharviour is correct?)
        self.event_served = 0

        # If we have enough device the training will be done, otherwise we wait other event
        # -1 is introduced because in the list is considered also the local worker, which is our coordinator
        if (len(self.server._known_workers) - 1) >= self.training_lower_bound:
            
            # if len(workers_training >= up):
            #     # TODO apply a selection criteria
            
            
            # Do the training
            print("I will do the training")
            
            # In this case Local the data will be created by us and then distributed
            federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader 
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
                .federate(tuple(self.server._known_workers.values())), # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
                batch_size=self.args.batch_size, shuffle=True)

            print(self.server._known_workers)
            
            

            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)
            models = {}
            for worker in list(self.server._known_workers.keys()):
                temp_model, loss = cf.train_local( worker=worker,
                model=self.model, opt=optimizer, epochs=self.args.epochs, federated_train_loader=federated_train_loader, args=self.args)
                models[worker] = temp_model


            print(models)

            # to_process = []
            # for key in models.keys():
            #     to_process.append(models[key])
            
            self.model = utils.federated_avg(models)
            print(self.model)
            # If we have enough worker I can delete all the known_workers, after the training
            self.__remove_safely_known_workers()
        else:
            # TODO Decide the correct behaviour: continue? Or throw everithing away?
            print("something") 

    def __remove_safely_known_workers(self, key=None):
        # Delete everithing exept me
        if key == None:
            for key in list(self.server._known_workers.keys())[1:]:
                del self.server._known_workers[key]
        else:
            del self.server._known_workers[key]


def main(argv):
    host = "localhost"
   
    keepalive = 60
    port = 1883
    topic = None

    

    try:
        opts, args = getopt.getopt(argv, "h:k:p:t:v",
                                   ["host","keepalive", "port",  "topic"])
    except getopt.GetoptError as s:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--host"):
            host = arg
        elif opt in ("-k", "--keepalive"):
            keepalive = int(arg)
        elif opt in ("-p", "--port"):
            port = int(arg)
        elif opt in ("-t", "--topic"):
            topic = arg
            print(topic)

    if topic == None:
        print("You must provide a topic to clear.\n")
        sys.exit(2)
    
    mqttc = Coordinator(10)
    rc = mqttc.run(host, port, topic, keepalive)
    # print("rc: "+str(rc))

if __name__ == "__main__":
    main(sys.argv[1:])
