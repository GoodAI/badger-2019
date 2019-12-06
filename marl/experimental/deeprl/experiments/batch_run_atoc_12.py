import os

root_cmd = "python marl/experimental/deeprl/experiments/run_multiagent.py"

for k in range(5):
    os.system(root_cmd + f" --type ATOC --name ATOC20-{k} --num-agents 12 --num-landmarks 12 --num-steps 350000")