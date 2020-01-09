#!/usr/bin/python

# TODO Make the inference asynchronous
# TODO Delete the print and add a logger
# TODO What happen if the coordinator receives event during a training phase?
# TODO Menage the error cases in training phase: (1) remove all known worker for training or (2) recall the method and try to avoid the error
# TODO websocket Server side has a problem when the connection is closed: It continous to wait from the same client, even if the connection has been closed. There is a way to stop the process?
# TODO Close the socket after the training (not in the client_federated file but here)
# TODO Check the torch.jit.trace
# TODO Manage the reception of an event during the training: now we have the error in the following line: start_loop = lambda : asyncio.new_event_loop().run_until_complete(self.__starting_training_remote())

# Be aware of these cases:
# 1) What happen if a new device desires to do the training after the window?
# We have some problems in case of local training, this because it is sequential. So, this device is not considered or the timer is restarted (it depends on when the self.event_served variable is changed).
# For example, if we do it at the beginning a new time window will be opened (problem of data distribution), if we
# do it at the end of the time window, all the event received during the training are not managed (this is the default behaviour). The last case, could generate some errors in the 
# training phase because the data is not distributed in these devices.
# In the remote case instead, the remote worker, for which the training is started, is deleted from the dict of devices to train.
# So, when new events from new devices are received during the training phase, they are considered after that the training already started will end.
# To avoid the loosing of model upgrade propagation, the self.event_served is changed at the end of the first window. So, the devices received during the
# training phase related to this window, which desire to be trained,  will be managed with the new model generated in from the previous training phase. 
# Of course, this requires that a new device will trigger the window opening.
# For example, at the end of the window_1 execution (so when the training of the devices received in this window start), three devices sent an event
# where they are ready to do the "TRAINING". The training of these three devices will start, only after the ending of the training related to the window_1. But,
# to trigger the execution of a new window_2, which will trigger the training of the previous three devices, another event from another devices must be received. This is how it works now, maybe some
# improvements is needed (see the graph on the notebook to understand better the problem)

# BROKER SETTINGS (First stop the default service launchctl stop homebrew.mxcl.mosquitto)
# /usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf --> launch mosquitto
# mosquitto_pub -t topic/state -m "(192.168.1.7, TRAINING)" --> event to publish
# In case of remote worker use the command described in the file

# How to run the coordinator:
# python federated_coordinator -t "topic/state" # local case
# python federated_coordinator -t "topic/state" -r # remote case


import sys
import getopt
import torch
import syft as sy
import asyncio
from torch import optim
import time
import os.path
from syft.frameworks.torch.federated import utils
from syft.workers.websocket_client import WebsocketClientWorker

import paho.mqtt.client as mqtt
from torchvision import datasets, transforms # datasets is used only to do some tests
from event_parser import EventParser
import client_federated as cf
import settings
from datasets import NetworkTrafficDataset, ToTensorLong, ToTensor
from threading import Timer

# This is important to exploit the GPU if it is available
use_cuda = torch.cuda.is_available()

# Seed for the random number generator
torch.manual_seed(1)

device = torch.device("cuda" if use_cuda else "cpu")

# @sy.func2plan(args_shape=[(1,1)])
# def inference_with_anomaly_detection(x):
#     print("nothing")
#     if x == 1.0:
#         print("Anomaly detected")

class Arguments():
    def __init__(self):
        self.batch_size = 10
        self.test_batch_size = 1024
        self.epochs = 1 # Remember to change the number of epochs
        self.federate_after_n_batches = -1 # In this way it will not be stopped during the training
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False
        self.test_path = "/Users/angeloferaudo/Downloads/UNSW_2018_IoT_Botnet_Final_10_best_Training_2.csv" # Insert the path for the testing evaluation

class Coordinator(mqtt.Client):
    def __init__(self, window, remote):
        super(Coordinator, self).__init__()
        settings.init()
        self.training_lower_bound = 1
        self.training_upper_bound = 100
        self.training_workers_id = []
        self.event_served = 0
        self.__hook = sy.TorchHook(torch)
        self.__hook.local_worker.is_client_worker = False # Set the local worker as a server: All the other worker will be registered in __known_workers
        self.server = self.__hook.local_worker
        self.window = window
        self.remote = remote
        self.path = './test.pth' # Where the model is saved, and from is loaded
        self.args = Arguments()
        


    def on_connect(self, mqttc, obj, flags, rc):
        print("Connected!")

    def on_message(self, mqttc, obj, msg):
        print(msg.topic+" "+str(msg.qos)+" "+str(msg.payload))
        parser = EventParser(msg.payload)
        
        # Obtain ip address
        ip_address = parser.ip_address()
     
        worker = None
        if not self.remote:
            # In case of local testing the event syntax is a bit different: e.g.
            # "(192.168.1.3,TRAINING)"
            print("Local testing")

            # Obtain the state of the server
            state = parser.state(local=True)
            if ip_address != -1 and state != None:
                worker = sy.VirtualWorker(self.__hook, ip_address)
            else:
                print("Ip address or state not valid") 

        else:
            print("Remote execution")
            state = parser.state()
            port = parser.port()
            # Create remote Worker
            if port != -1 and ip_address != -1 and state != None:
                identifier = ip_address + ":" + str(port)
                print("Remote worker idetifier: " + identifier)
                kwargs_websocket = {"host": ip_address, "hook": self.__hook, "verbose": True}
                worker = WebsocketClientWorker(id=identifier, port=port, **kwargs_websocket)
            else:
                print("Server worker: syntax event error")
        
            
        if worker != None:
            # TRAINING
            if state == "TRAINING":
                self.event_served += 1
                settings.training_devices[worker.id] = worker # registration

                # [print(worker1[1]) for worker1 in self.server._known_workers.items() if worker1[0] != 'me']
                print(worker)
                
                if self.event_served == 1: 
                    # Start the timer after received an event. This creates our window
                    print("Timer starting")

                    # We have two different method, one for the virtual worker and one for the remote
                    if self.remote:
                        start_loop = lambda : asyncio.new_event_loop().run_until_complete(self.__starting_training_remote())
                        t = Timer(self.window, start_loop)
                        t.start()

                    else:
                        t = Timer(self.window, self.__starting_training)
                        t.start()
                    
            elif state == "INFERENCE":
                if self.remote:
                    # Infernce is implemented only for remote case, because is useless in the local case
                    # Remeber LOCAL is only to test that everything work properly
                    

                    # The model is loaded each time that someone asks for it
                    # In this case, we don't need to save it at the end
                    if not os.path.exists(self.path): # If the model doesn't exist we create a new one
                        model = cf.TestingRemote()
                    else:
                        print('Loading model')
                        model = cf.TestingRemote()
                        model.load_state_dict(torch.load(self.path))
            
                    # Obtain pointer to the data
                    data_pointer = worker.search("inference")# This return a list, we take only the first element
                    print(data_pointer)
                    if data_pointer != []:
                        data_pointer = data_pointer[0]
                        
                        # Send the model
                        model = model.send(worker)

                        # To avoid waste of bandwidth the best solution is to create a behaviour that send back an event in case of anomaly
                        # Build the plan
                        # inference_with_anomaly_detection.build()
                        # pointer_plan = inference_with_anomaly_detection.send(worker)

                        # Apply the model to the data
                        with torch.no_grad():
                            # TODO Define a behaviour in case of anomaly detected!
                            output_pointer = model(data_pointer)
                            prediction_pointer = output_pointer.argmax(1, keepdim=True)
                            print(prediction_pointer.get())

                        del self.server._known_workers[worker.id]

                        # After the inference, close the ws with the server
                        worker.close()
                    else:
                        print("Inference data not found!")
                else:
                    print("Inference not implemented for local purpose")
        
            elif state == "NOT_READY":
                # This method is useful in training case only
                print(ip_address + " is not ready anymore, removing from the known lists")
                if remote:
                    identifier = ip_address + ":" + str(port)
                else:
                    identifier = ip_address
                del settings.training_devices[identifier]
                del self.server._known_workers[identifier]

            else:
                print("No behavior defined for this event")
            
            print("Training devices: " + str(settings.training_devices))
            print("All known workers: " + str(self.server._known_workers))
        else:
            print("Some problems occurred")

    def on_publish(self, mqttc, obj, mid):
        print("mid: "+str(mid))


    def run(self, host, port, topic, keepalive):
        self.connect(host, port)
        self.subscribe(topic, 0)

        print("Coordinator started. Press CTRL-C to stop")
        try:
            self.loop_forever()
        except KeyboardInterrupt:
            print("Coordinator stopped.")
    

    # The local case is useful only for testing purposes
    def __starting_training(self):
        
        # If we have enough device the training will be done, otherwise we wait other event
        # -1 is introduced because in the list is considered also the local worker, which is our coordinator
        if len(settings.training_devices) >= self.training_lower_bound:
            
            if len(settings.training_devices)>= self.training_upper_bound:
            #     # TODO apply a selection criteria
                print("To implement")
        
            
            # The model must be loaded each time that we do the training
            if not os.path.exists(self.path): # If the model doesn't exist we create a new one
                model = cf.Net()
            else:
                model = cf.Net()
                model.load_state_dict(torch.load(self.path))
      
            # Do the training
            print("Local training start")
            
            # In this case Local the data will be created by us and then distributed (This is the MNIST example)
            federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader 
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
                .federate(tuple(settings.training_devices.values())), # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
                batch_size=self.args.batch_size, shuffle=True)

            # Test loader to evaluate our model
            test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
                batch_size=self.args.test_batch_size, shuffle=True)

            # Optimizer used Stochstic gradient descent
            optimizer = optim.SGD(model.parameters(), lr=self.args.lr)
            models = {}
            # Be aware: in this case the training is sequential (It is not important to have asynchronism in this case)
            for worker in list(settings.training_devices.keys()):
                temp_model, loss = cf.train_local( worker=worker,
                model=model, opt=optimizer, epochs=1, federated_train_loader=federated_train_loader, args=self.args)
                models[worker] = temp_model


            print(models)
            
            # Apply the federated averaging algorithm
            model = utils.federated_avg(models)
            print(model)
            
            # After the training we save the model
            torch.save(model.state_dict(), self.path)

            # Evaluate the model obtained
            cf.evaluate_local(model=model, args=self.args, test_loader=test_loader, device=device)
            # If we have enough worker I can delete all the known_workers, after the training
            self.__remove_safely_known_workers(training=True)

            print("Workers after training: " + str(self.server._known_workers()))


        else:
            # TODO Decide the correct behaviour: continue? Or throw everithing away?
            print("something")

        self.event_served = 0

    async def __starting_training_remote(self):
        # TODO Solve this problem: Returning the object as is.
        # warnings.warn('The input to trace is already a ScriptModule, tracing it is a no-op. Returning the object as is.')
        # I think that is due to the fact that the model is still serialized, so to solve this copy the weight in another
        # model or save them someqhere and then reload them!

        if len(settings.training_devices) >= self.training_lower_bound:
            
            if len(settings.training_devices) >= self.training_upper_bound:
                # TODO apply a selection criteria
                print("To implement")
            
            # The model must be loaded each time that we do the training
            if not os.path.exists(self.path): # If the model doesn't exist we create a new one
                print("No existing model")
                # Settinf common hyperparameters
                # input_dim = 10 #Here typically they use the shape
                # output_dim = 1
                # n_layers = 2

                # model = cf.GRUModel(input_dim, 10, output_dim, n_layers)
                # hidden = model.init_hidden(self.args.batch_size)
                # test_seq = torch.LongTensor(1,10).to(device)
                model = cf.TestingRemote2()
                model = model.float() # I don't know if this is correct, but with this the method works
                # model = cf.TestingRemote()
            else:
                model = cf.TestingRemote2()
                model.load_state_dict(torch.load(self.path))

            print("Remote method")
            self.event_served = 0
            # Remember that the serializable model requires a starting point:
            # for this reason we pass the mockdata: torch.zeros([1, 1, 28, 28]
            # traced_model = torch.jit.trace(model, torch.zeros(1, 2))
            traced_model = torch.jit.trace(model, torch.zeros(10))
            learning_rate = self.args.lr
            
            # Schedule calls for each worker concurrently:
            results = await asyncio.gather( 
                *[
                    cf.train_remote(
                        worker=worker[1],
                        traced_model=traced_model,
                        batch_size=self.args.batch_size,
                        optimizer="SGD",
                        max_nr_batches=self.args.federate_after_n_batches,
                        epochs=self.args.epochs,
                        lr=learning_rate,
                    )
                    for worker in settings.training_devices.items() # maybe now it doesn't require the index (worker[1])
            ]
            )
            models = {}
            loss_values = {}

            # Federate models (note that this will also change the model in models[0]
            for worker_id, worker_model, worker_loss in results:
                if worker_model is not None:
                    models[worker_id] = worker_model
                    loss_values[worker_id] = worker_loss

                # When the training is ended this remote worker can be removed from the devices to be trainind
                del self.server._known_workers[worker_id]
            print(models) # Logging purposes

            # Apply the federated averaging algorithm
            model = utils.federated_avg(models)
            print(model) # Logging purposes
            print("After training: Training devices " + str(settings.training_devices))
            print("After training: Known workers " + str(self.server._known_workers))
            # print(self.server._objects) # Logging purposes
            
            # After the training we save the model 
            torch.save(model.state_dict(), self.path)

            # Evaluation of the model
            test_dataset = NetworkTrafficDataset(self.args.test_path, transform=ToTensor())
            test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)
            cf.evaluate(test_loader,device)

            # Window restart
            self.event_served = 0
        
        else:
            # TODO Create a behaviour when we are under the threshold
            print("something")

    def __remove_safely_known_workers(self, key=None, training=False):
        # This method is used only for local purpose
        if key == None:
            if training:
                # In case of training I have to delete remove from the known worker only the one that requested for training
                for key in list(self.server._known_workers.keys())[1:]:
                    if key in settings.training_devices.keys():
                        del self.server._known_workers[key]
                settings.training_devices = {}
            else:
                # In the other case, I have to delete all the worker except the training worker
                for key in list(self.server._known_workers.keys())[1:]:
                    if len(settings.training_devices) != 0:
                        if not key in settings.training_devices.keys():
                            del self.server._known_workers[key]
                    else:
                        del self.server._known_workers[key]
        else:
            del self.server._known_workers[key]
            

    def _ciao():
        print("LA MADONNA DI POMPEI")
        # identifier = ip_address + ":" + str(port)
        # kwargs_websocket = {"host": ip_address, "hook": self.__hook, "verbose": True}
        # worker = WebsocketClientWorker(id=identifier, port=port, **kwargs_websocket)
        # return worker
        # if self.remote:
        #     # REMOTE CASE
        #     if port != -1:
        #         identifier = ip_address + ":" + str(port)
        #         print("Remote worker idetifier: " + identifier)
        #         print("we are here")
        #         kwargs_websocket = {"host": ip_address, "hook": self.__hook, "verbose": True}
        #         worker = WebsocketClientWorker(id=identifier, port=port, **kwargs_websocket)
        #     else:
        #         print("Server worker: " + ip_address + " port not valid!")
        #     return worker
        # else:
        #     # LOCAL CASE
        #     # In this case we have only the training phase
        #     # Create the worker and register it 
        #     worker = sy.VirtualWorker(self.__hook, ip_address) 
        #     settings.training_devices[worker.id] = worker 
        #     return worker # In this case is not important have a return statement

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
    
    mqttc = Coordinator(2, remote)
    mqttc.run(host, port, topic, keepalive)
    # print("rc: "+str(rc))

if __name__ == "__main__":
    main(sys.argv[1:])