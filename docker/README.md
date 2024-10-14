## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), but we give some
quick links here:

* [OSX](https://docs.docker.com/installation/mac/): [docker toolbox](https://www.docker.com/toolbox)
* [ubuntu](https://docs.docker.com/installation/ubuntulinux/)

For GPU support, install compatible NVIDIA drivers with CUDA9.0 and CUDNN 7.6

## Running the container

Build the container:

    $ docker build -f Dockerfile -t scdd_isk .

## Build the docker
Use the modified Dockerfile to build the docker
## Making graphical interface available
To have the graphical interface available and capable of displaying all the images from the dataset and the predictions first run:
```
xhost +local:docker
```
## Running the docker
After building the docker run the docker with the following command to have a shared folder with the host and also the graphical interface available.

```
sudo docker run -v /home/shiva/Documents/code/SCDD-image-segmentation-keras/share:/SCDD-image-segmentation-keras/share -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --env QT_X11_NO_MITSHM=1 -it -v ~/.ssh:/root/.ssh scdd_isk
```
