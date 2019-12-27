import re
# Typical string (ip_address, state)
# TODO add more states (for example when a client is not available anymore)
states = ["TRAINING", "INFERENCE", "NOT_READY"]
# TODO solve the space problem

class EventParser():
    def __init__(self, message):
        # The message received is typically encoded in binary
        # So, must be transformed in string
        self.message = message.decode('utf-8').replace(" ","")

    
    def ip_address(self):
        
        # Now the string must be parsed, in order to obtain the ip address of the device
        # 1) remove the brackets
        to_parse = re.sub(r'[\(\)]', '', self.message)

        # 2) obtain the ip address
        ip_address = re.split(r',', to_parse)[0]

        # 3) verify ip address
        pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        if(re.match(pattern, ip_address)):
            return ip_address
        else:
            return -1
    
    def port(self, local=False):
        if local:
            return -1
        
        # 1) remove the brackets
        to_parse = re.sub(r'[\(\)]', '', self.message)

        # 2) obtain the port
        port = int(re.split(r',', to_parse)[1])

        # 3) Verify the port
        if port in range(65535):
            return port
        else:
            return -1

    def state(self, local=False):
        # Now the string must be parsed, in order to obtain the event sent by the device (which represent its state)
        # 1) remove the brackets
        to_parse = re.sub(r'[\(\)]', '', self.message)

        # 2) obtain the state
        if local:
            state = re.split(r',', to_parse)[1]
        else:
            state = re.split(r',', to_parse)[2]
        
        if state in states:
            return state
        else:
            return None
    
    def training(self):
        return states[0]
    
    def inference(self):
        return states[1]


