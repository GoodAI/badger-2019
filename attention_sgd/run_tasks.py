import os

repeats = 1

params = \
         ['epochs=5000 key_size=3 n_experts=20 task_size=20 attention_beta=5.0'] * repeats + \
         []
common = ''

for param in params:
    merged_param = f'{common} {param}'
    print(f'Running with params {merged_param}')
    os.system(f'python search_experiment.py with {merged_param}')
