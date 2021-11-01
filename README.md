Final results! https://www.youtube.com/watch?v=nqZ4wrig3UM

# neural-video-generator
Generates videos with VQGAN+CLIP inside docker as a standalone task

- [Build container](#build-container)
- [IO Contract](#io-contract)
  - [STATUS File](#status-file)
- [Run Job](#run-job)
- [Run Simulated Job](#run-simulated-job)

<p align="middle">
  <img src="https://user-images.githubusercontent.com/16694980/133938520-349133fb-b531-4a30-a0ac-c9c5da79e677.jpg" width="40%" />
  <img src="https://user-images.githubusercontent.com/16694980/133938653-a652c272-28b7-40cd-9c99-528b2c694e62.gif" width="40%" />
</p>
<p align="middle">
  <b>Seed image + Input prompt:</b> "dmt trip a million eyeballs hyper-realistic"
</p>

## Build container
```bash
cd neural-video-generator
docker build . -t neural-video-generator
```

## IO Contract
Users must setup an **IO directory** that contains an **input directory** before running a job. This directory is mounted as a volume inside the container to handle IO to and from the host machine. 

The container automatically handles storing and clearing items in the output directory along with giving status of currently executing jobs in the textfile `STATUS`
```
sample_IO/
├── input            <-- User required to create & set up contents of this directory before execution
│   ├── 0.jpg        <-- Seed image
│   └── config.json  <-- Job configuration (num frames, output filenames, etc.)
├── output
│   ├── frames
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── 0003.png
│   └── my_output_video.mp4
└── STATUS
```
### STATUS File
Status file is periodically updated by the container and can be polled to query the current state of a job. The status file will be used to determine if a job has is starting, completed, failed, or is in progress. 

#### Starting state 
Job will breifly be in a starting state when a new job is kicked off.  

Template
```
# STARTING [unique id]
```
Example contents of STATUS file
```
STARTING 1234567890
```

#### In Progress state 
Template
```
# IN_PROGRESS [unique id] [% completed] [ETA in seconds]
```
Example contents of STATUS file
```
IN_PROGRESS 1234567890 45% 15s
```
#### Completed state 
Template
```
# COMPLETED [unique id] [MP4 filename]
```
Example contents of STATUS file
```
COMPLETED 1234567890 1234567890-dmt_trip_a_million_eyeballs_hyper-realistic.mp4
```
#### Failed state 
Template
```
# FAILED [unique id]
```
Example contents of STATUS file
```
FAILED 1234567890
```

## Run Job
Warning! Be sure host has at least 11 GB free memory! This is actually executing the neural networks on your host. 
```bash
cd neural-video-generator
docker run --rm -it -v $PWD/sample_IO:/video_io neural-video-generator python3 exec.py --mode=PROD
```
The sample job generates 30 frames with the network by default, this can be changed in `sample_IO/config.json` 

Output is stored at `sample_IO/output/my_output_video.mp4`

## Run Job With GPU
First install `nvidia-docker2`

[Ubuntu setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
Run job with `--gpus all` flag
```
docker run --gpus all --rm -it -v $PWD/sample_IO:/video_io neural-video-generator python3 exec.py --mode=PROD
```

## Run Simulated Job
This is so integration can be done on hosts without GPU access or enough resources for CPU execution. A fake job will run and `.mp4` output will be stored in the IO directory. 
```bash
cd neural-video-generator
docker run --rm -it -v $PWD/sample_IO:/video_io neural-video-generator python3 exec.py --mode=TEST
```
Change duration of the simulated run by adding the flag  `--test_duration=10`

Test failure case by changing mode to `--mode=TEST_FAIL`. This simulates what the contents of the `STATUS` file would look like during a failure.
