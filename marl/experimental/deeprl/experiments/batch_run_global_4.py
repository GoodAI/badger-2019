import os

root_cmd = "python marl/experimental/deeprl/experiments/run_multiagent.py"


for k in range(5):
    os.system(root_cmd + f" --type GLOBAL --name GLOBAL4-{k} --num-agents 4 --num-landmarks 4 --num-steps 2000000")