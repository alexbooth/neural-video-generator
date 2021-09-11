# neural-video-generator
Generates videos with VQGAN+CLIP inside docker as a standalone task

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
Status file is periodically updated by the container and can be polled to query the current state of a job. The status file will be used to determine if a job has completed, failed, or is in progress. 

TODO. need to define & implement this

## Build container
```bash
cd neural-video-generator
docker build . -t neural-video-generator
```

## Run Job
Warning! Be sure host has at least 11 GB free memory! This is actually executing the neural networks on your host. 
```bash
cd neural-video-generator
docker run --rm -it -v $PWD/sample_IO:/video_io neural-video-generator python3 exec.py --mode=PROD
```
The sample job generates 3 frames with the network by default, this can be changed in `sample_IO/config.json` 

Output is stored at `sample_IO/output/my_output_video.mp4`

## Run Simulated Job
This is so integration can be done on hosts without GPU access. A fake job will run and `.mp4` output will be stored in the IO directory. 
```bash
cd neural-video-generator
docker run --rm -it -v $PWD/sample_IO:/video_io neural-video-generator python3 exec.py --mode=TEST
```
Change duration of the simulated run by adding the flag  `--test_duration=10`

Test failure case by changing mode to `--mode=TEST_FAIL`. This simulates what the contents of the `STATUS` file would look like during a failure. ##TODO IMPLEMENT THIS
