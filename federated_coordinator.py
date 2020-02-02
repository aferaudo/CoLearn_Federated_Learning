# TODO websocket Server side has a problem when the connection is closed: It continous to wait from the same client, even if the connection has been closed. There is a way to stop the process?
# TODO Check the torch.jit.trace
# TODO Which is the behaviour when the connection is lost? It continues to wait? It is an error, we have to manage it
# TODO Complete closing when ctrl-c is called: It is just missing the closing of the window thread
# TODO Do the federated after n_batches in a more clever way

# Be aware of these cases:
# 1) What happen if a new device desires to do the training after the window? --> solution of the todo
# We have some problems in case of local training, this because it is sequential. So, this device is not considered or the timer is restarted (it depends on when the settings.event_served variable is changed).
# For example, if we do it at the beginning a new time window will be opened (problem of data distribution), if we
# do it at the end of the time window, all the event received during the training are not managed (this is the default behaviour). The last case, could generate some errors in the 
# training phase because the data is not distributed in these devices.
# In the remote case instead, the remote worker, for which the training is started, is deleted from the dict of devices to train.
# So, when new events from new devices are received during the training phase, they are considered after that the training already started will end.
# To avoid the loosing of model upgrade propagation, the settings.event_served is changed at the end of the first window. So, the devices received during the
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

# BE CAREFUL: The program works only with the version 0.2.0a2, further version will be supported as soon as possible
# TODO If you have some problem, reinstall the old version of syft syft-0.2.0a2 (current version: syft-0.2.2a1)
import argparse
import logging
import psutil

import sys
import torch
import syft as sy
import asyncio
from torch import optim
import time
import os.path
from syft.frameworks.torch.federated import utils # Old version
# from syft.frameworks.torch.fl import utils
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


class Arguments():
    def __init__(self):
        self.batch_size = 1
        self.test_batch_size = 1024
        self.epochs = 1 # Remember to change the number of epochs
        # federated_after_n_batches: number of training steps performed on each remote worker before averaging
        self.federate_after_n_batches = -1 # In this way it will not be stopped during the training
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False
        self.test_path = "/Users/angeloferaudo/Downloads/UNSW_2018_IoT_Botnet_Final_10_best_Training_0.csv" # Insert the path for the testing evaluation
    
    def set_federated_batches(self, batches):
        self.federate_after_n_batches = batches

# Arguments
parser = argparse.ArgumentParser(description="Run Federated coordinator")
parser.add_argument(
    "--port", "-p", type=int, default=1883, help="port number of the where the broker is listining (default 1883)"
)
parser.add_argument("--host", type=str, default="localhost", help="broker ip address (default localhost)")

parser.add_argument(
    "--topic", "-t", type=str, required=True, help="topic where the event must be published"
)
parser.add_argument(
    "--remote", "-r", action='store_true', help="Remote learning activation"
)
parser.add_argument(
    "--window", "-w", type=int, default=1, help="temporal window size (default 1)"
)
parser.add_argument(
    "--encryption", "-e", action='store_true', help="Simulates the encryption on two virtual workes (you have to generate two workers with mosquitto_pub)"
)
parser.add_argument(
    "--federated_round", "-f", type=int, default=1, help="(work in progress) Enable or disable the rounds, if the round are activated the training is stopped after nbatches and then restarted (round must be greater then one)"
)

class Coordinator(mqtt.Client):
    def __init__(self, window, remote, federated_round, encryption):
        """
        This class is a client mqtt which waits for mqtt events in order to train and do inference on IoT devices
        Args:
            window: define the window dimension (collection of IoT devices before to start the training) in secs
            remote: enable the Coordinator for remote training (if not specified the training is done in local only for test purpose)
            federated_round: enabling the training in round, instead of having the training on all the data this will be done on 1000 data at time
        """
        super(Coordinator, self).__init__()
        settings.init()
        
        # Command parameters
        self.window = window
        self.remote = remote
        self.enabled_round = federated_round
        self.encryption = encryption

        # Other useful parameters
        self.training_lower_bound = 1
        self.training_lower_bound_enc = 1
        self.training_upper_bound = 100
        self.training_workers_id = []
        self.__hook = sy.TorchHook(torch)
        self.__hook.local_worker.is_client_worker = False # Set the local worker as a server: All the other worker will be registered in __known_workers
        self.server = self.__hook.local_worker
        self.local_thread = None
        self.log_interval = 1 # Log info at each batch
        self.path = './test.pth' # Where the model is saved, and from is loaded
        self.args = Arguments()
        


    def on_connect(self, mqttc, obj, flags, rc):
        logging.info("Connected!")

    def on_message(self, mqttc, obj, msg):
        logging.info(msg.topic+" "+str(msg.qos)+" "+str(msg.payload))
        parser = EventParser(msg.payload)
        
        # Obtain ip address
        ip_address = parser.ip_address()
        worker = None
        
        if not self.remote:
            # In case of local testing the event syntax is a bit different: e.g.
            # "(192.168.1.3,TRAINING)"
            logging.info("Local testing")

            # Obtain the state of the server
            state = parser.state(local=True)
            
            # Create Local worker
            if ip_address != -1 and state != None and state != "NOT_READY":
                worker = sy.VirtualWorker(self.__hook, ip_address)
            elif state == "NOT_READY":
                logging.info("NOT_READY state received")
            else:
                logging.info("Ip address or state not valid") 

        else:
            logging.info("Remote execution")
            state = parser.state()
            port = parser.port()
            
            # Create remote Worker
            if port != -1 and ip_address != -1 and state != None and state != "NOT_READY":
                identifier = ip_address + ":" + str(port)
                logging.info("Remote worker idetifier: " + identifier)
                kwargs_websocket = {"host": ip_address, "hook": self.__hook, "verbose": True}
                try:
                    worker = WebsocketClientWorker(id=identifier, port=port, **kwargs_websocket)
                except:
                    e = sys.exc_info()[0]
                    logging.info("Error " + str(e))

            elif state == "NOT_READY":
                logging.info("NOT_READY state received")
            else:
                logging.info("Server worker: no worker gerated")
        
            
        if worker != None or state == "NOT_READY":
            # TRAINING
            if state == "TRAINING":
                settings.event_served += 1
                settings.training_devices[worker.id] = worker # registration

                # [print(worker1[1]) for worker1 in self.server._known_workers.items() if worker1[0] != 'me']
                logging.info(worker)
                
                if settings.event_served == 1: 
                    # Start the timer after received an event. This creates our window
                    logging.info("Timer starting")

                    # We have two different method, one for the virtual worker and one for the remote
                    if self.remote:
                        # start_loop = lambda : asyncio.new_event_loop().run_until_complete(self.__starting_training_remote())
                        start_loop = lambda : asyncio.new_event_loop().run_until_complete(training_remote(
                                                                                                        lower_bound=self.training_lower_bound, 
                                                                                                        upper_bound=self.training_upper_bound,
                                                                                                        path=self.path,
                                                                                                        args=self.args,
                                                                                                        general_known_workers=self.server._known_workers,
                                                                                                        round=self.enabled_round
                                                                                                        ))
                        self.local_thread = Timer(self.window, start_loop)
                        self.local_thread.start()

                    else:
                        if self.encryption:
                            function_to_start = lambda : starting_training_enc(
                                                                                lower_bound=self.training_lower_bound_enc,
                                                                                upper_bound=self.training_upper_bound,
                                                                                path=self.path,
                                                                                args=self.args,
                                                                                server=self.server,
                                                                                hook=self.__hook
                                                                                )
                        else:
                            function_to_start = lambda : starting_training_local(
                                                                                lower_bound=self.training_lower_bound_enc,
                                                                                upper_bound=self.training_upper_bound,
                                                                                path=self.path,
                                                                                args=self.args,
                                                                                server=self.server
                                                                                )

                        t = Timer(self.window, function_to_start)
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
                        logging.info('Loading model')
                        model = cf.TestingRemote()
                        model.load_state_dict(torch.load(self.path))
            
                    # Obtain pointer to the data
                    data_pointer = worker.search("inference")# This return a list, we take only the first element
                    logging.info(data_pointer)
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
                            logging.info(prediction_pointer.get())

                        del self.server._known_workers[worker.id]

                        # After the inference, close the ws with the server
                        worker.close()
                    else:
                        logging.info("Inference data not found!")
                else:
                    logging.info("Inference not implemented for local purpose")
        
            elif state == "NOT_READY":
                # This method is useful in training case only
                logging.info(ip_address + " is not ready anymore, removing from the known lists")
                if self.remote:
                    identifier = ip_address + ':' + str(port)
                else:
                    identifier = ip_address

                if identifier in settings.training_devices.keys():
                    if self.remote:
                        worker = settings.training_devices.get(identifier)
                        logging.info("Worker to remove: " + str(worker))
                        worker.close() # Close the connection in remote case
                    del settings.training_devices[identifier]
                    del self.server._known_workers[identifier]

            else:
                logging.info("No behavior defined for this event")
            
            logging.info("ONMESSAGE: Training devices: " + str(settings.training_devices))
            logging.info("ONMESSAGE: All known workers: " + str(self.server._known_workers))
        else:
            logging.info("Some problems occurred")

    def on_publish(self, mqttc, obj, mid):
        logging.info("mid: "+str(mid))

    def run(self, host, port, topic):
        self.connect(host, port)
        self.subscribe(topic, 0)

        logging.info("Coordinator started. Press CTRL-C to stop")
        try:
            self.loop_forever()
        except KeyboardInterrupt:
            logging.info("Coordinator stopped.")
            if self.local_thread != None:
                logging.info("Cancelling the timer..")
                self.local_thread.cancel() # This works only if the timer objects is in a waiting phase, otherwise thr training will go ahead in any case
                for key in self.server._known_workers.keys():
                    if key == "me":
                        pass
                    else:
                        worker = settings.training_devices[key]
                        logging.info("Closing socket for worker " + str(worker))
                        worker.close()
                logging.info("Done")
    
   


def starting_training_local(lower_bound, upper_bound, path, args, server):
    
    if len(settings.training_devices) >= lower_bound:
            
        if len(settings.training_devices)>= upper_bound:
            logging.info("Applying selection criteria")

        # Copy virtual workers to train 
        to_train = settings.training_devices.copy()

        # Loading of the model
        model = cf.FFNN()
        logging.info("Loading model procedure started...")
        if not os.path.exists(path): # If the model doesn't exist we create a new one
            logging.info("No existing model")               
            # model = model.float() 
        else:
            logging.info("Loading of the model..")
            model.load_state_dict(torch.load(path))
            logging.info("Model loading successfull")
        logging.info("Done")

        # For logging purpose: It will be commented in future version
        for param in model.parameters():
            print(param)
        

        # Distribution data among the virtual workers
        logging.info("Distribution data among the virtual workers")
        federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader 
                NetworkTrafficDataset(args.test_path, ToTensor())
                .federate(tuple(to_train.values())), # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
                batch_size=args.batch_size, shuffle=True)
        
        logging.info("Done")

        # Optimizer creation
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        models = {}

        
        # Be aware: in this case the training is sequential (It is not important to have asynchronism in this case)
        logging.info("Start training")
        for worker in list(to_train.keys()):
            logging.info("Training for worker: " + str(worker))
            temp_model, loss = cf.train_local(worker=worker,
            model=model, opt=optimizer, epochs=1, federated_train_loader=federated_train_loader, args=args)
            models[worker] = temp_model
       
        logging.info("End training")

        
        logging.info(models)
        
        # Apply the federated averaging algorithm
        model = utils.federated_avg(models)
        logging.info(model)

        # For logging purpose: It will be commented in future version
        # for param in model.parameters():
        #     print(param)
        
        # save model
        # torch.save(model.state_dict(), self.path)

        # Deleting workers from training list
        for worker in to_train.keys():
            logging.info("Removing: " + str(worker) + " from training devices")
            del settings.training_devices[worker]
            del server._known_workers[worker]
        
        logging.info("End training local")
    
    # Restarting window
    settings.event_served = 0

# TODO In a second moment it could not work, manage this problem
def starting_training_enc(lower_bound, upper_bound, path, args, server, hook):
    """This function simulates the encryption of the model: It works only with two virtual workes"""
    
    if len(settings.training_devices) >= lower_bound:
        
        if len(settings.training_devices) >= upper_bound:
           logging.info("Applying selection criteria")
        
        # Copy virtual workers to train (MAX 2) (this is useful so then I can delete only the devices that have been trained)
        to_train = {}
        i = 0
        for worker in settings.training_devices.keys():
            if i >= 2:
                break
            else:
                to_train[worker] = settings.training_devices[worker]
                i += 1

        # Loading of the model
        model = cf.FFNN()
        logging.info("Loading model procedure started...")
        if not os.path.exists(path): # If the model doesn't exist we create a new one
            logging.info("No existing model")               
            model = model.float() # I don't know if this is correct, but with this the method works
            
        else:
            logging.info("Loading of the model..")
            model.load_state_dict(torch.load(path))
            # for param in model.parameters():
            #     print(param.data)
            logging.info("Model loading successfull")

        logging.info("Done")

        # Creation new cryptoprovider
        crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

        # Data distribution
        logging.info("Distribute the data among the virtual workers...")
        n_train_items = 1000
        private_train_loader = cf.get_private_data_loaders(workers=to_train, precision_fractional=3, crypto_provider=crypto_provider, args=args, n_train_items=n_train_items)
        logging.info("Done")

        # Distribution of the encrypted model among the workers (fixed_precision is needed, in order to perform consistently operations like the weight update)
        logging.info("Encryption and distribution of the model...")
        model = model.fix_precision().share(*to_train, crypto_provider=crypto_provider, requires_grad=True)
        print(model)
        logging.info("Done")

        # Optimizer creation and fix precision on it
        logging.info("Optimizer creation and fix precision on it...")
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        optimizer = optimizer.fix_precision()
        logging.info("Done")
        
        # Start training
        logging.info("Start training")
        for i in range(args.epochs):
            cf.encrypted_training(model=model, optimizer=optimizer, epoch=i, private_train_loader=private_train_loader, args=args)
        logging.info("Done")
        
        # Printing new model parameters
        model = model.get().float_precision()
        
        # for param in model.parameters():
        #         print(param.data)
        
        # Save model


        # Deleting workers from training list
        for worker in to_train.keys():
            logging.info("Removing: " + str(worker) + " from training devices")
            del settings.training_devices[worker]
            del server._known_workers[worker]
        logging.info("End encryption")

    else:
        logging.info("No behaviour defined for a number of workers less than " + str(lower_bound))

    # Restarting window
    settings.event_served = 0


async def training_remote(lower_bound, upper_bound, path, args, general_known_workers, round):
    """Set all the parameters and start the remote training
    Args:
        lower_bound: minum number of devices required for training
        upper_bound: maximum number of devices required for training
        path: path where the model could be stored
        args: argument for the training settings
        general_known_workers: all the known workers of the caller
        round: is an integer which enables to do n cycles of training, instead of training on all the data in one time
        encrypted: start the encrypted training
    Returns:
        no return
    """
    try:
        if len(settings.training_devices) >= lower_bound:
            
            if len(settings.training_devices) >= upper_bound:
                # TODO apply a selection criteria: This depends on the execution environment
                logging.info("Applying selection criteria")
            

            # Copy the devices to train (In this way all the other device training will start after that this training is ended)
            to_train = {}
            to_train = settings.training_devices.copy() #Otherwise, if a device is added to the list during the training, is possible that it will do the training for a lower number of round

            # Loading model
            # In the hierchical architecture we will ask the model for that category of device
            # In this case we have only one model
            model = cf.FFNN()
            logging.info("Loading model procedure started")
            if not os.path.exists(path): # If the model doesn't exist we create a new one
                logging.info("No existing model")               
                model = model.float() # I don't know if this is correct, but with this the method works

                for param in model.parameters():
                    print(param.data)
            else:
                logging.info("Found a model..")
            
                model.load_state_dict(torch.load(path))

                for param in model.parameters():
                    print(param.data)
                
            logging.info("Done")
            

            logging.info("Obtain the traced model...")
            traced_model = model.get_traced_model()
            logging.info("Done")

            learning_rate = args.lr
            # Remember that the serializable model requires a mock object
            traced_model = model.get_traced_model()
            

            # Schedule calls for each worker concurrently:
            if round > 1:
                logging.info("Round activated!")
                args.set_federated_batches(10000)
            
            logging.info("Federated batches: " + str(args.federate_after_n_batches))

            # Round start
            for i in range(round):
                logging.info("\n\n#### ROUND {} #####".format(i))
                logging.info("Remote training on multiple devices started...")
                results = await asyncio.gather( 
                    *[
                        cf.train_remote(
                            worker=worker[1],
                            traced_model=traced_model,
                            batch_size=args.batch_size,
                            optimizer="SGD",
                            max_nr_batches=args.federate_after_n_batches,
                            epochs=args.epochs,
                            lr=learning_rate,
                        )
                        for worker in to_train.items() # maybe now it doesn't require the index (worker[1])
                ]
                )
                models = {}
                loss_values = {}
                logging.info("Remote training on multiple devices ended")

                # Federate models (note that this will also change the model in models[0]
                for worker_id, worker_model, worker_loss in results:
                    if worker_model is not None:
                        models[worker_id] = worker_model
                        logging.info("Loss for worker id: " + str(worker_id) + " " + str(worker_loss))
                    
                print(models) # Logging purposes
                

                # Apply the federated averaging algorithm
                traced_model = utils.federated_avg(models) # Maybe here I've to use the traced_model
                print(model) # Logging purposes
            # Round end

            # Logging purpose to verify if the parameters are changed
            # for param in model.parameters():
            #         print(param.data)

            # Close all the sockets and delete the workers
            for worker_id, _, _ in results:
                logging.info("Closing socket for " + str(worker_id))
                worker = settings.training_devices[worker_id]
                worker.close()

                logging.info("Deleting " + str(worker_id) + " from known worker")
                del settings.training_devices[worker.id]
                del general_known_workers[worker_id]
            logging.info("Done")
            

            # After the training we save the model 
            torch.save(traced_model.state_dict(), path)

            # Evaluation of the model
            # test_dataset = NetworkTrafficDataset(args.test_path, transform=ToTensor())
            # test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)
            # cf.evaluate(traced_model,test_loader,device)
            
        else:
            # TODO define a possible behavior
            logging.info("No behaviour defined for the number of devices achieved")
    finally:
        # Window restart
        settings.event_served = 0


def main(argv):
    # Model to test instance cration
    # model = cf.GRUModel(input_dim=10, hidden_dim=10, output_dim=1, n_layers=1)
    format = "%(asctime)s: %(message)s"

    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.info(os.getpid())
    mqttc = Coordinator(args.window, args.remote, args.federated_round, args.encryption)
    mqttc.run(args.host, args.port, args.topic)
    # model = cf.FFNN()
    # model.load_state_dict(torch.load('./test.pth'))
    # for param in model.parameters():
    #     print(param.data)
    # test_dataset = NetworkTrafficDataset("/Users/angeloferaudo/Downloads/UNSW_2018_IoT_Botnet_Final_10_best_Training_3.csv", transform=ToTensor())
    # test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)
    # cf.evaluate(model,test_loader,device)

if __name__ == "__main__":

    args = parser.parse_args()
    main(args)