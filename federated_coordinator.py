#!/usr/bin/python
# TODO client side you can realize the publisher. This means that when the 
# start_worker.py is launched we have also to specify, if is launched for 
# training or for inference. Thus, we can manage the envent that will be sent.

# BROKER SETTINGS (First stop the default service launchctl stop homebrew.mxcl.mosquitto)
# /usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf --> launch mosquitto
# mosquitto_pub -t topic/state -m "(192.168.1.7, TRAINING)" --> event to publish

# Important things to understand:
# If two workers have the same id at the end only one will be created

import sys
import getopt
import torch
import syft as sy
import asyncio
from torch import optim
import time
from syft.frameworks.torch.federated import utils
from syft.workers.websocket_client import WebsocketClientWorker

import paho.mqtt.client as mqtt
from torchvision import datasets, transforms # datasets is used only to do some tests
from event_parser import EventParser
import client_federated as cf
from threading import Timer

# This is important to exploit the GPU if it is available
use_cuda = torch.cuda.is_available()

# Seed for the random number generator
torch.manual_seed(1)

device = torch.device("cuda" if use_cuda else "cpu")


class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 100
        self.federate_after_n_batches = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False

class Coordinator(mqtt.Client):

    def __init__(self, window, remote):
        super(Coordinator, self).__init__()
        self.training_lower_bound = 2
        self.training_upper_bound = 100
        self.event_served = 0
        self.__hook = sy.TorchHook(torch)
        self.__hook.local_worker.is_client_worker = False # Set the local worker as a server: All the other worker will be registered in __known_workers
        self.server = self.__hook.local_worker
        self.window = window
        self.remote = remote
        # TODO load the model from a file if it is present, otherwise create a new one
        self.model = cf.TestingRemote()
        self.args = Arguments()
        


    def on_connect(self, mqttc, obj, flags, rc):
        print("Connected!")

    def on_message(self, mqttc, obj, msg):
        print(msg.topic+" "+str(msg.qos)+" "+str(msg.payload))
        parser = EventParser(msg.payload)
        
        # Obtain ip address
        ip_address = parser.ip_address()
     
        # Obtain the state of the server
        state = parser.state()

        # Obtain the port for the remoteworker (Server)
        port = parser.port()
        
        if ip_address != -1 and state != None:
            print(ip_address + " " + state + " " + str(port))
            identifier = ip_address + ":" + str(port)
            print(identifier)
            if state == "TRAINING":
                self.event_served += 1
                
                if not self.remote:
                    # Create Virtual Worker
                    worker = sy.VirtualWorker(self.__hook, ip_address) # registration
                else:
                    # Create remote Worker
                    print("Remote")
                    if port != -1:
                        kwargs_websocket = {"host": ip_address, "hook": self.__hook, "verbose": True}
                        worker = WebsocketClientWorker(id=identifier, port=port, **kwargs_websocket)
                    else:
                        print("Server worker: " + ip_address + " port not valid!")

                [print(worker1[1]) for worker1 in self.server._known_workers.items() if worker1[0] != 'me']
                print(worker)
                
                if self.event_served == 1: 
                    # Start the timer after received an event. This creates our window
                    print("Timer starting")

                    # We have two different method, one for the virtual worker and one for the remote
                    if self.remote:
                        print("Remote execution")
                        t = Timer(self.window, self.__starting_training_remote)
                        # asyncio.get_event_loop().run_until_complete(self.__starting_training_remote())
                        t.start()
                        
                    else:
                        t = Timer(self.window, self.__starting_training)
                        t.start()
                    
                    

            elif state == "INFERENCE":
                # TODO inference
                print("Behavior inference")
            
            elif state == "NOT_READY":
                # TODO When a client is no more available we have to remove it from the workers list (training, inference or both?)
                self.__remove_safely_known_workers(key=ip_address)
                print(self.server._known_workers)

            else:
                print("No behavior defined for this event")

            


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
            
            if len(self.server._known_workers >= self.training_upper_bound):
            #     # TODO apply a selection criteria
                print("To implement")
            
            
            # Do the training
            print("I will do the training")
            
            # In this case Local the data will be created by us and then distributed (This is the MNIST example)
            federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader 
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
                .federate(tuple(self.server._known_workers.values())), # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
                batch_size=self.args.batch_size, shuffle=True)

            # Test loader to evaluate our model
            test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
                batch_size=self.args.test_batch_size, shuffle=True)

            # Optimizer used Stochstic gradient descent
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)
            models = {}
            for worker in list(self.server._known_workers.keys()):
                temp_model, loss = cf.train_local( worker=worker,
                model=self.model, opt=optimizer, epochs=self.args.epochs, federated_train_loader=federated_train_loader, args=self.args)
                models[worker] = temp_model


            print(models)
            
            # Apply the federated averaging algorithm
            self.model = utils.federated_avg(models)
            print(self.model)

            # Evaluate the model obtained
            cf.evaluate_local(model=self.model, args=self.args, test_loader=test_loader, device=device)
            # If we have enough worker I can delete all the known_workers, after the training
            self.__remove_safely_known_workers()


        else:
            # TODO Decide the correct behaviour: continue? Or throw everithing away?
            print("something")




    def __starting_training_remote(self):
        # TODO Implement the remote training
        # TODO Solve this problem: Returning the object as is.
        # warnings.warn('The input to trace is already a ScriptModule, tracing it is a no-op. Returning the object as is.')
        # I think that is due to the fact that the model is still serialized, so to solve this copy the weight in another
        # model or save them someqhere and then reload them!
        print("Remote method")
        self.event_served = 0
        # Remember that the serializable model requires a starting point:
        # for this reason we pass the mockdata: torch.zeros([1, 1, 28, 28]
        traced_model = torch.jit.trace(self.model, torch.zeros(1, 2))
        learning_rate = self.args.lr
        
        print("Start fitting...")
        results = [
                cf.train_remote(
                    worker=worker[1],
                    traced_model=traced_model,
                    batch_size=self.args.batch_size,
                    optimizer="SGD",
                    max_nr_batches=self.args.federate_after_n_batches,
                    epochs=self.args.epochs,
                    lr=learning_rate,
                )
                for worker in self.server._known_workers.items() if worker[0] != 'me'
        ]
        print("Fitting ended!")
        models = {}
        loss_values = {}

        # Federate models (note that this will also change the model in models[0]
        for worker_id, worker_model, worker_loss in results:
            if worker_model is not None:
                models[worker_id] = worker_model
                loss_values[worker_id] = worker_loss
        
        print(models)

        # Apply the federated averaging algorithm
        self.model = utils.federated_avg(models)
        print(self.model)

        # If we have enough worker I can delete all the known_workers, after the training
        self.__remove_safely_known_workers()



    def __remove_safely_known_workers(self, key=None):
        # Delete everithing exept me
        if key == None:
            for key in list(self.server._known_workers.keys())[1:]:
                # TODO try the following code
                # self.server.remove_worker_from_local_worker_registry(key) 
                del self.server._known_workers[key]
        else:
            del self.server._known_workers[key]


def main(argv):
    host = "localhost"
   
    keepalive = 60
    port = 1883
    topic = None
    remote = False
    

    try:
        opts, args = getopt.getopt(argv, "h:k:p:t:vr",
                                   ["host","keepalive", "port",  "topic","remote"])
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
        elif opt in ("-r", "--remote"):
            remote = True

    if topic == None:
        print("You must provide a topic to clear.\n")
        sys.exit(2)
    
    mqttc = Coordinator(20, remote)
    rc = mqttc.run(host, port, topic, keepalive)
    # print("rc: "+str(rc))

if __name__ == "__main__":
    main(sys.argv[1:])
