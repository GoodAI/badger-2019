import os

root_cmd = "python run_multiagent.py"

for k in range(5):
    os.system(root_cmd + f" --type ATOC --name ATOC4-{k} --num-agents 4 --num-landmarks 4 --num-steps 2000000")