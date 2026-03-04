## Verify Dataset Integrity

Before preprocessing, verify the integrity of the downloaded dataset:

```bash
cd dataset
md5sum -c checksums.md5
```

If you see the following, the dataset files are complete:

```bash
SN_Dataset.zip: OK
TT_Dataset.zip: OK
```

To extract the dataset:

```bash
unzip TT_Dataset.zip
unzip SN_Dataset.zip
```

Note: This repository ships a cleaned dataset (irrelevant or redundant files have been removed). 

## Preprocess the Raw Dataset
***Since the original repository does not provide the complete raw data preprocessing code, this fork attempts to fill in the missing puzzle pieces and fix some bugs in the original code.***
```
pip install drain3
export ROOT_PATH=path/to/the/downloaded/eadro/dataset
cd codes/preprocess/
python parse_raw_records.py
python parse_raw_metrics.py 
python parse_raw_traces.py
python parse_raw_logs.py
python align.py --name TT
python align.py --name SN
```

## Train and Test the Model
```bash
# Return to the codes directory
cd ../

# Run on the TT dataset
python main.py --data TT

# Run on the SN dataset
python main.py --data SN
```

## Running Logs (with default hyper parameters)
Because the codebase dependency version is a bit out of date, I have to run the model on CPU for now.
### SN-Dataset
``` 
......
2025-09-24 20:33:56,451 P228100 INFO Epoch 48/50, training loss: 0.05156 [0.61s]
2025-09-24 20:33:57,018 P228100 INFO Epoch 49/50, training loss: 0.04530 [0.57s]
2025-09-24 20:33:58,958 P228100 INFO Test -- F1: 0.9802, Rec: 0.9920, Pre: 0.9687, HR@1: 0.9679, ndcg@1: 0.9679, HR@3: 0.9920, ndcg@3: 0.9826, HR@5: 0.9920, ndcg@5: 0.9826
2025-09-24 20:33:59,584 P228100 INFO Epoch 50/50, training loss: 0.03268 [0.62s]
2025-09-24 20:33:59,584 P228100 INFO * Best result got at epoch 49 with HR@1: 0.9679
2025-09-24 20:33:59,584 P228100 INFO Current hash_id 78e573d7
```

### TT-Dataset
```
......
2025-09-24 22:00:03,676 P379891 INFO Epoch 48/50, training loss: 0.07763 [5.95s]
2025-09-24 22:00:09,854 P379891 INFO Epoch 49/50, training loss: 0.06530 [6.18s]
2025-09-24 22:00:28,727 P379891 INFO Test -- F1: 0.9644, Rec: 0.9631, Pre: 0.9656, HR@1: 0.9613, ndcg@1: 0.9613, HR@3: 0.9631, ndcg@3: 0.9624, HR@5: 0.9631, ndcg@5: 0.9624
2025-09-24 22:00:34,795 P379891 INFO Epoch 50/50, training loss: 0.07146 [6.07s]
2025-09-24 22:00:34,795 P379891 INFO * Best result got at epoch 9 with HR@1: 0.9876
2025-09-24 22:00:34,796 P379891 INFO Current hash_id 0aba95d5
```

<img width="200" alt="截屏2022-09-19 下午9 50 34" src="https://user-images.githubusercontent.com/112700133/191033061-ea4a1671-26c7-4d52-b3ed-3495a2ae0292.png">

![](https://img.shields.io/badge/version-0.1-green.svg) 

****
Artifacts accompanying the paper *Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data* published at ICSE 2023. 
This tool try to model the intra- and inter-dependencies between microservices for troubleshooting, enabling end-to-end anomaly detection and root cause localization.

<img width="400" alt="dependency_00" src="https://user-images.githubusercontent.com/112700133/191036446-d4cf8d07-bd4e-4452-a3e2-f7d4e9da0624.png">

## Data
Our data are at https://doi.org/10.5281/zenodo.7615393.


## Dependencies
`pip install -r requirements.txt`

## Run
`cd codes & python main.py --data <dataset>`


## Architecture
![Eadro](https://user-images.githubusercontent.com/49298462/217256928-f0d61857-678b-4456-a024-359326a2c45d.png)

## Folder Structure
```
.
├── README.md
├── codes                                             
│   ├── base.py                                         traing and test
│   ├── main.py                                         lanuch the framework
│   ├── model.py                                        the main body (model) of the work
│   ├── preprocess                                      data preprocess                
│   │   ├── align.py                                    align different data sources according to the time
│   │   ├── single_process.py                           handle each data source individually
│   │   └── util.py
│   └── utils.py
├── requirements.txt
└── structure.txt
```

## UI
The final visualized page should be like:
<img width="1919" alt="截屏2023-02-07 下午9 28 22" src="https://user-images.githubusercontent.com/49298462/217257747-e53afafe-ea3f-4024-8760-34d0963a863d.png">

## Concact us
🍺 Feel free to leave messages in "Issues"! 
