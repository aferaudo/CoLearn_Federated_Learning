# TODO Specify in the thesis that this implementation doesn't have any kind of logic! The logic will be realized in future versions-
# How to use example:
# python remote_worker.py --host 127.0.0.1 -p 8778 -b localhost -t "topic/state" -w 1 -e "TRAINING" --verbose <-dt> <data available for training> <-di> <data available for inference>
# THE HOST (--host) MUST BE SPECIFIED AS AN IP (also for localhost communication), instead no problem for the broker

import argparse

import torch as th
from syft.workers.websocket_server import WebsocketServerWorker
import paho.mqtt.client as mqtt
import syft as sy
from threading import Timer
import numpy as np

from torchvision import datasets, transforms # datasets is used only to do some tests
from datasets import NetworkTrafficDataset, ToTensor, Normalize


# Arguments
parser = argparse.ArgumentParser(description="Run websocket server worker.")
parser.add_argument(
    "--port", "-p", type=int, default=8777, help="port number of the websocket server worker, e.g. --port 8777"
)
parser.add_argument("--host", type=str, required=True, help="host for the connection: represent the ip address of the network interface where the communication will happen")

parser.add_argument(
    "--broker", "-b", type=str, required=True, help="Broker of the mqtt protocol"
)
parser.add_argument(
    "--topic", "-t", type=str, required=True, help="topic where the event must be published"
)
parser.add_argument(
    "--wait", "-w", type=int, default=5, help="Number of second to wait before to send the event"
)
parser.add_argument(
    "--event", "-e", type=str, default="TRAINING", help="state of the client (TRAINING, INFERENCE, NOT_READY), e.g. --event TRAINING"
)
parser.add_argument(
    "--training", "-dt", type=str, default=None, help="data training path. This will be mandatory in future versions"
)
parser.add_argument(
    "--inference", "-di", type=str, default=None, help="data inference path. This will be mandatory in future versions"
)
parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="if set, websocket server worker will be started in verbose mode",
)


# This script creates a worker and populate it with some toy data using worker.add_dataset, the dataset is identified by a key in this case xor.
def main(args):  # pragma: no cover
    hook = sy.TorchHook(th)
    identifier = args.host + ":" + str(args.port)
    kwargs = {
        "id": identifier,
        "host": args.host,
        "port": args.port,
        "hook": hook,
        "verbose": args.verbose,
        # "cert_path": "/Users/angeloferaudo/Desktop/Unibo Magistrale/Tesi/mud_file_server/mudfs/certs/server.pem", # Insert the cert here
        # "key_path": "/Users/angeloferaudo/Desktop/Unibo Magistrale/Tesi/mud_file_server/mudfs/certs/server.key", # Insert the key here
    }

    # Create a client object
    client = mqtt.Client("woker")

    # Connect to the broker
    client.connect(args.broker)

    # String to publish

    to_publish = '('+ args.host + ', ' + str(args.port) +', ' + args.event +')'
    

    if args.training == None:
        # Setup toy data
        data = th.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True)
        target = th.tensor([[1.0], [1.0], [0.0], [0.0]], requires_grad=False)
        # Create a dataset using the toy data
        dataset = sy.BaseDataset(data, target)
    else:
        batch_size = 3 
        print(args.training)
        apply = transforms.Compose([ToTensor()])
        dataset = NetworkTrafficDataset(args.training, transform=apply)

    
    # dataloader = th.utils.data.DataLoader(dataset, shuffle=True)
    # for data, target in dataloader:
    #     print("DATA: " + str(data))
    #     print("TARGET: " + str(target))
    
    if args.inference == None:
        # Setup toy data
        y = th.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]],requires_grad=False).tag("inference")
    else:
        # TODO change with the real dataset
        y = th.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]],requires_grad=False).tag("inference")

    # Create websocket worker
    worker = WebsocketServerWorker(data=[y], **kwargs)
    
    # Tell the worker about the dataset
    worker.add_dataset(dataset, key="training")

    fn = lambda : client.publish(args.topic, to_publish)
    # Publish the event that the server is ready after an interval
    t = Timer(args.wait, fn)
    t.start()

    # Start worker
    worker.start()




if __name__ == "__main__":

    args = parser.parse_args()
    main(args)