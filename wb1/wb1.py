from dateutil import parser
import requests
import yaml
import logging
requests.packages.urllib3.disable_warnings()
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
import json
import base64
import pprint
from datetime import datetime, timezone
import os
import time
from subprocess import Popen, PIPE
import boto3
import glob

HOME_DIR = os.path.dirname(os.path.abspath(__file__))
pp = pprint.PrettyPrinter()
with open('secrets.yml', 'r') as f:
    secrets = yaml.safe_load(f.read())
auth = (secrets['api_user'], secrets['api_pass'])
s = requests.Session()
s.auth = auth
s.verify = False


class WB1:
    def __init__(self):
        self.submission_for_training = ""
        self.current_jobdir = ""
        self.s3_bucket_name = "workerbae-3-test-bucket"
        self.s3_client = boto3.client('s3')

    def runner(self):
        while True:
            if self.get_oldest_submitted() == False:
                continue
            self.download_training_data()
            self.generate_config_json()
            self.set_state_training()
            self.run_docker()
            self.upload_to_s3()
            self.set_state_complete()
            os.chdir(HOME_DIR)

    def get_oldest_submitted(self):
        print('Checking for oldest submitted')
        submissions = json.loads(s.get('https://bitmover.solutions:8443/list').text)
        now = datetime.now(timezone.utc)
        oldest = now
        oldest_submission = ""
        for submission in submissions:
            this_ts = parser.parse(submission['ts'] + " UTC")
            if submission['state'] == 'submitted' and this_ts < oldest:
                oldest = this_ts
                oldest_submission = submission
                print('yolo')
        if oldest == now:
            print('Doing nothing')
            time.sleep(5)
            return False
        else:
            pp.pprint(self.submission_for_training)
            self.submission_for_training = oldest_submission

    def download_training_data(self):
        print(f"Download submission data for {self.submission_for_training['submission_id']}")
        THIS_JOB_DIR=f"{self.submission_for_training['submission_id']}"
        os.makedirs(THIS_JOB_DIR)
        os.chdir(THIS_JOB_DIR)
        os.makedirs("input")
        os.makedirs("output")
        # Save the submission's data
        payload = {
            "submission_id": self.submission_for_training['submission_id']
        }
        download = s.post('https://bitmover.solutions:8443/download', data=json.dumps(payload))
        download_json = json.loads(download.text)

        with open(f"{self.submission_for_training['submission_id']}.jpg", 'wb') as f:
            try:
                output = base64.b64decode(download_json['image'].replace('data:image/jpeg;base64,',''))
            except Exception as e:
                print(e)
                raise e
            f.write(output)
        with open(f"{self.submission_for_training['submission_id']}.txt", 'w') as f:
            f.write(self.submission_for_training['texts'])


    def generate_config_json(self):
        print(f"Generating config for {self.submission_for_training['submission_id']}")
        config = {
            "prompt": self.submission_for_training['texts'],
            "init_image": f"{self.submission_for_training['submission_id']}.jpg",
            "unique_id": self.submission_for_training['submission_id'],
            "num_frames": 300,
        }
        with open('./input/config.json', 'w') as f:
            f.write(json.dumps(config))
        #os.chmod('./input/config.json', 0o777)

    def set_state_training(self):
        print(f"Setting state=training for {self.submission_for_training['submission_id']}")
        payload = {
            "submission_id": self.submission_for_training['submission_id'],
            "field": "state",
            "value": "training",
        }
        s.post('https://bitmover.solutions:8443/update', data=json.dumps(payload))

        payload = {
            "submission_id": self.submission_for_training['submission_id'],
            "field": "ts",
            "value": datetime.now(timezone.utc),
        }
        s.post('https://bitmover.solutions:8443/update', data=json.dumps(payload,indent=4, sort_keys=True, default=str))
    
    def run_docker(self):
        print(os.getcwd())
        p = Popen([])
        run_command = f"docker run --rm -v {os.getcwd()}:/video_io neural-video-generator python3 exec.py --mode=PROD"
        p = Popen(run_command.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, error = p.communicate()
        print(output.decode('utf-8'))
        print(error.decode('utf-8'))
        print(p.returncode)
    
    def set_state_complete(self):
        print(f"Setting state=complete for {self.submission_for_training['submission_id']}")
        payload = {
            "submission_id": self.submission_for_training['submission_id'],
            "field": "state",
            "value": "complete",
        }
        s.post('https://bitmover.solutions:8443/update', data=json.dumps(payload))

        payload = {
            "submission_id": self.submission_for_training['submission_id'],
            "field": "ts",
            "value": datetime.now(timezone.utc),
        }
        s.post('https://bitmover.solutions:8443/update', data=json.dumps(payload,indent=4, sort_keys=True, default=str))

    def upload_to_s3(self):
        mp4 = glob.glob('output/*.mp4')
        if len(mp4) == 0:
            print('ERROR, weird, no mp4 file found??')
            return False
        
        self.s3_client.upload_file(mp4[0], self.s3_bucket_name, os.path.basename(mp4[0]))
        return

wb1 = WB1()
wb1.runner()
