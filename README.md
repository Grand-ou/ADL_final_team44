# ADL22_Final_Team44
GitHub repo for ADL final project.
# seen
For testing seen recommending, run bash as follows
## course
```bash
cd "$(git rev-parse --show-toplevel)/seen"

bash ./run.sh --pred_course
```
## topic
```bash
cd "$(git rev-parse --show-toplevel)/seen"

bash ./run.sh --pred_topic
```
# unseen
## course
For testing unseen course recommending task, run bash as follows
```bash
cd unseen/topic

bash download.sh

bash run.sh
#predict file will save as pred_3.csv
```
## course
For testing unseen course recommending task, run bash as follows
```bash
cd unseen/course

bash download.sh
# for example !bash test.sh hahow/data data
# ckiptagger will read data as ws = WS(path_to_ckip_data, disable_cuda=False)
bash test.py path_to_hahow_data path_to_ckip_data
#predict file will save as courses_submission_unseen.csv
```
