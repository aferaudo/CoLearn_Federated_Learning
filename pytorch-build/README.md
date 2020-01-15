# Step-by-Step procedure to install pytorch 1.3.0 and pysyft on raspberry pi 3+

In this guide the processor architecture version used is **armv7l**. For different version you have to find the suitable wheel file for your architecture. Try to use the link in the file *pytorch-arm-builds.txt*.

**N.B** To find the processor architecture type the following command:

```
uname -a
```

## Installing PyTorch dependencies

Update the package lists

```
sudo apt-get update
```

Install the dependencies

```
sudo apt install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-dev python3-yaml python3-setuptools
```

## Install pytorch

Copy the file in the rapberry, you can use git and download the file directly or any kind of copying procedure. I will use *scp*.

```
scp torch-filename.whl pi@ip_address:/home/pi
```

Now you can install pytorch

```
sudo pip3 install torch-filename.whl # This operation requires some time(5 minutes)
```

If you have a very light version of the operating system you need to install pip3

```
sudo apt install python3-pip
```

Try if everything work

```
python3

import torch
```

Maybe you can have some dependecies problem. In my case, I needed also numpy because it has never been installed on the raspberry :)

```
sudo pip3 install numpy
```

After this I had also this error

```
libf77blas.so.3: cannot open shared object file: No such file or directory
```

You can solve it by installing the library

```
sudo apt-get install libatlas-base-dev
```

## Install syft

Here there are the problem.

Considering that syft has a lot of dependencies and that pytorch is not installed directly, this because at the time of writing of this guide doesn't exist an official package, we need to install syft without dependencies.

This means that after the installation, we need to install them manually

```
sudo pip3 install syft --no-dependencies
```

Now we need the dependencies. To see what dependencies you need, simply run

```
python3

import syft
```

If you have the following error, you need to install the tf_encrypted package (this depend on the version of syft that you have installed)

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.7/dist-packages/syft/__init__.py", line 7, in <module>
    from syft import dependency_check
  File "/usr/local/lib/python3.7/dist-packages/syft/dependency_check.py", line 18, in <module>
    tfe_spec = importlib.util.find_spec("tf_encrypted")
AttributeError: module 'importlib' has no attribute 'util
```

```
sudo pip3 install tf_encrypted
```

After that will be asked to install *tblib*

```
sudo pip3 install tblib
```

Now you will have all the packages required. In my case are the following:

```
sudo pip3 install Flask flask-socketio lz4 msgpack websocket-client websockets zstd
```

**Note:** Ignore the pytorch package, because we have already installed it (consequently even torchvision)

At this point you will have everything installed

## Install sklearn

I had some problems also in this procedure.
The problem in this case is the scipy installation. During this procedure, it is possible that the fortran library is missing. So, being based on fortran and C library, the procedure will be stopped and then reversed.
To solve the problem first of all we need to install the fortran library.

```
sudo apt-get install gfortran
```

However, I still have the problem. After some research on the raspeberry forum, I found that the wheel file provided are not well defined.
So, the procedure to install scipy is the following

```
sudo apt-get install python3-scipy
```

Then you can install sklearn

```
pip3 install sklearn
```


## Other tutorials

This guide was realised only because I couldn't find a well explained guide to install pytorch 1.3 and the last version of syft for the arm architecture. I needed this version to support the remote workers (WebsocketServerWorker).

So, I think that could be useful for you also this link:

- [link1](https://medium.com/secure-and-private-ai-writing-challenge/a-step-by-step-guide-to-installing-pytorch-in-raspberry-pi-a1491bb80531)
- [link2](https://medium.com/secure-and-private-ai-writing-challenge/a-step-by-step-guide-to-installing-pysyft-in-raspberry-pi-d8d10c440c37)
- [link3](https://blog.openmined.org/federated-learning-of-a-rnn-on-raspberry-pis/) 