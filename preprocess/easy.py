#!/usr/bin/env python3
import json
import os
import numpy as np

if os.getcwd() != "/home/catalan/Desktop/6.8611/project/K-BERT/preprocess/qa/ARC-V1-Feb2018-2/ARC-Challenge":
    os.chdir("/home/catalan/Desktop/6.8611/project/K-BERT/preprocess/qa/ARC-V1-Feb2018-2/ARC-Challenge")
with open("ARC-Challenge-Train.jsonl", "r") as f:
    train_list = list(f)
with open("ARC-Challenge-Dev.jsonl", "r") as f:
    dev_list = list(f)
with open("ARC-Challenge-Test.jsonl", "r") as f:
    test_list = list(f)

list_json_data = [train_list, dev_list, test_list]
output_names = ["ARC-Challenge-Train.tsv", "ARC-Challenge-Dev.tsv", "ARC-Challenge-Test.tsv"]
for d,name in zip(list_json_data, output_names):
    # Columns: question id, question, answer, label (0 or 1)
    num_cols = 4
    # max_num_a = 4 # some questions have 3 choices
    # out_data = np.empty((max_num_a*len(d), num_cols), dtype=str)
    out_data = []
    for question_json in d:
        question_dict = json.loads(question_json)
        q = question_dict['question']['stem']
        q_id = question_dict['id']
        a = {ans['label']:ans['text'] for ans in question_dict['question']['choices']}
        correct_ans = a[question_dict['answerKey']]
        out_data.append([q_id, q, correct_ans, "1"])

        del a[question_dict['answerKey']]
        n_incorrect = len(a)
        incorrect_ans = list(a.values())
        num_to_pick = np.random.choice(np.arange(1, n_incorrect + 1), p=np.full((n_incorrect,), 1.0/n_incorrect))
        add_answers = []
        for _ in range(num_to_pick):
            i = np.random.choice(np.arange(len(incorrect_ans)))
            add_answers.append(incorrect_ans[i])
            incorrect_ans.pop(i)
        for a in add_answers:
            out_data.append([ q_id, q, a, "0"])
    np.savetxt(name, out_data, delimiter="\t", fmt="%s")
    print(f"Saved successfully to {name}")