import argparse

import torch as th
from syft.workers.websocket_server import WebsocketServerWorker
import paho.mqtt.client as mqtt
import syft as sy
from threading import Timer

# Arguments
parser = argparse.ArgumentParser(description="Run websocket server worker.")
parser.add_argument(
    "--port", "-p", type=int, default=8777, help="port number of the websocket server worker, e.g. --port 8777"
)
parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
parser.add_argument(
    "--id", type=str, default="bob", help="name (id) of the websocket server worker, e.g. --id alice"
)
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
    "--verbose",
    "-v",
    action="store_true",
    help="if set, websocket server worker will be started in verbose mode",
)

def publish_event(broker, topic, event):
    # Create a client object
    client = mqtt.Client("woker")

    # Connect to the broker
    client.connect(broker)

    # Publish the envet
    client.publish(topic, event)


# This script creates a worker and populate it with some toy data using worker.add_dataset, the dataset is identified by a key in this case xor.
def main(args):  # pragma: no cover
    hook = sy.TorchHook(th)
    kwargs = {
        "id": args.id,
        "host": args.host,
        "port": args.port,
        "hook": hook,
        "verbose": args.verbose,
    }

    # Create a client object
    client = mqtt.Client("woker")

    # Connect to the broker
    client.connect(args.broker)

    # String to publish
    # TODO obtain automatically the ip address
    to_publish = '('+ '192.168.1.1, ' + args.event + ')'
    # Setup toy data (xor example)
    data = th.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True)
    target = th.tensor([[1.0], [1.0], [0.0], [0.0]], requires_grad=False)

    x = th.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]],requires_grad=False).tag("classify")

    # Create websocket worker
    worker = WebsocketServerWorker(data=[x], **kwargs)

    # Create a dataset using the toy data
    dataset = sy.BaseDataset(data, target)

    # Tell the worker about the dataset
    worker.add_dataset(dataset, key="xor1")

    fn = lambda : client.publish(args.topic, to_publish)
    # Publish the event that the server is ready after an interval
    t = Timer(args.wait, fn)
    t.start()

    # Start worker
    worker.start()

    return worker


if __name__ == "__main__":

    args = parser.parse_args()
    main(args)