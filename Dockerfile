FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y &&                                        \
    apt-get install -y --no-install-recommends                  \
        apt-utils                                               \
        exempi                                                  \
        python3                                                 \
        python3-pip                                             \            
        git                                                     \
        curl                                                    \
        vim                                                     \
        ffmpeg  &&                                              \
        rm -rf /var/lib/apt/lists/* &&                          \
    pip3 install                                                \
        transformers                                            \
        ftfy                                                    \
        regex                                                   \
        tqdm                                                    \
        omegaconf                                               \
        pytorch-lightning                                       \ 
        kornia                                                  \
        einops                                                  \
        wget                                                    \
        pillow==7.1.2                                           \
        imageio-ffmpeg                                          \
        torchvision                                             \
        imageio

#ImageNet 16384 
RUN curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' && \
    curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'

RUN git clone https://github.com/alexbooth/neural-video-generator.git --recursive && \
    mv /vqgan_imagenet_f16_16384.ckpt /neural-video-generator/ &&                    \
    mv /vqgan_imagenet_f16_16384.yaml /neural-video-generator/

ENV VIDEO_IO_PATH=/video_io

#RUN python3 neural-video-generator/python/exec.py --mode SETUP

# TODO delete... local dev stuff
#COPY ./python/exec.py neural-video-generator/python/exec.py
#COPY ./python/utils.py neural-video-generator/python/utils.py
#WORKDIR neural-video-generator
