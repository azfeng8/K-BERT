#!/usr/bin/env python3

import numpy as np
import pandas as pd
path = './preprocess/conceptnet5/one_million_examples.tsv'

BANNED = {'arse', 'arsehead' 'asshole', 'arsehole', 'bastard', 'bitch', 'bullshit', 'crap', 'cunt', 'cock', 'cocksucker', 'chink'
        'damn', 'dick', 'dickhead', 'fuck', 'fucker', 'goddamn', 'horseshit', 'kike', 'motherfucker', 'piss', 'prick', 'pussy', 'shit', 'shite', 
        'shitter', 'slut', 'spastic', 'twat', 'wanker'
         }

df = pd.read_csv(path, sep="\t")
n_rows_before = df.shape[0]

for bad_word in BANNED:
    df = df[df['arg2'].str.contains(bad_word) == False]
n_rows_after = df.shape[0]
print(n_rows_before, n_rows_after)

df.to_csv("cleaned_kg.tsv", sep="\t", encoding="utf-8", index=False)