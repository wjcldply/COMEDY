import json
import random; random.seed(42)
from pprint import pprint
from tqdm import tqdm

# Load Dataset
with open('validation_data.json', 'r') as infile:
    raw = json.load(infile)

session_Ns = ['session2', 'session3', 'session4']
assert len(raw) == 3 and len(raw['session2']) == len(raw['session3']) == len(raw['session4']) == 1000

new = {}
for session_N in session_Ns:
    # Create New Dataset (Randomly Cutoff 'Current Session' between 30~70% Ranges)
    session_N_list = []  # list of dict (each dict: 'history_sessions', 'current_session_original', 'current_session_test'(cutoff randomly), 'id')
    for sample in raw[session_N]:  # sample = {'history_sessions', 'current_session', 'id}
        modified_sample = {}
        modified_sample['history_sessions'] = sample['history_sessions']
        modified_sample['current_session_original'] = sample['current_session']

        # Randomly Cutoff 'Current Session' 생성
        rand_cutoff_idx = random.choice([i for i in range(int(len(sample['current_session'])*0.3), int(len(sample['current_session'])*0.7)+1) if i%2==0])
        modified_sample['current_session_test'] = sample['current_session'][:rand_cutoff_idx+1]

        modified_sample['id'] = sample['id']
        session_N_list.append(modified_sample)

    assert len(session_N_list) == len(raw[session_N]) == 1000
    new[session_N] = session_N_list
with open('test_data.json', 'w') as outfile:
    json.dump(new, outfile, ensure_ascii=False, indent=4)