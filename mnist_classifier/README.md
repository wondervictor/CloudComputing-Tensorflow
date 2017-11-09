## MNIST Classifier - Round 1

### Usage

```
mkdir data
```

**1. Normal Version**

```python

python main.py --data_dir='./data'

```

**2. Distributed Version**


```python

# Parameter Server 1:

python distributed_classifier --job_name=ps --task_index=0 --ps_hosts=0.0.0.0:2333,0.0.0.0.0:2224 \
--worker_hosts=0.0.0.0:2344,0.0.0.0:2345 --data_dir=./data/

# Parameter Server 2:

python distributed_classifier --job_name=ps --task_index=1 --ps_hosts=0.0.0.0:2333,0.0.0.0.0:2224 \
--worker_hosts=0.0.0.0:2344,0.0.0.0:2345 --data_dir=./data/


# Worker 1

python distributed_classifier --job_name=worker --task_index=0 --ps_hosts=0.0.0.0:2333,0.0.0.0.0:2224 \
--worker_hosts=0.0.0.0:2344,0.0.0.0:2345 --data_dir=./data/

# Worker 2

python distributed_classifier --job_name=worker --task_index=1 --ps_hosts=0.0.0.0:2333,0.0.0.0.0:2224 \
--worker_hosts=0.0.0.0:2344,0.0.0.0:2345 --data_dir=./data/


```

