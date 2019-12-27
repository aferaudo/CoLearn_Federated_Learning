# This file was added to solve the problem of receiving event during the training phase.
# This file contains all the global variable needed.
# The list is: 
#   training_devices: is a dict which contains all the devices that must be managed for the training. This is important for the remote case,
#                    because it allows to remove the worker from the list when the training is already started (directly in the function train in client_federated). 
#                    So, the webesocket overlapping is avoided.

def init():
    global training_devices
    training_devices = {}