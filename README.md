# Federated Learning: distributed architecture for MUD compliant networks

The architecture provided is based on a pattern publish/subscribe. Particularly, the technology used is **MQTT**, which is suitable especially for IoT devices. In fact, the aim is to allow the IoT devices to signal their training/inference intention and automatically start the operation chosen. In order to enable the Federated Learning (FL) automation a Coordinator, which is able to coordinate all the FL operations is provided. Thus, an example of interaction between device and Coordinator is:

- the device publishes its training intention;

- the Coordinator (subscriber) receives the status change and wait for other devices (this can be useful to apply some selection criterias across the devices);

- After the waiting time (this term refers to the temporal window) expires, the model is sent to all (or some) devices collected and the training start;

- When the training end, the model updates are aggregated by using the Federated Averaging algorithm.

Thus, the most important concepts provided by the architecture are: **publish/subscribe model** for FL operations automation and **temporal winodw** for device collection.

Furthermore, [here](https://github.com/aferaudo/CoLearn_Federated_Learning/tree/master/data) are provided some data that results from experiments conducted on real devices (raspberry pi).

This architecture is designed to work together with MUD so that all the Federated Learning partecipant are **MUD compliant devices**.

The work has been accepted at EdgeSys 2020 conference. You can find the paper [here](https://github.com/aferaudo/CoLearn_Federated_Learning/tree/master/paper/EdgeSys2020.pdf) and slides [here](https://mega.nz/file/9RZC3Ypb#vBaHJMedH3kVetcrGW2EFxfqdaOZM-kCTmw7yNmch-Y). Please cite the paper as <Feraudo2020> ([bibitex](https://github.com/aferaudo/CoLearn_Federated_Learning/tree/master/paper/bibitex.txt)).