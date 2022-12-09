#!/usr/bin/env python3
"""
Author: Annie Feng

This file takes the first 1133083 parsed English entries in ConceptNet and saves to tsv format.
The tsv has 3 columns: relation 'rel', arg1 'arg1', arg2 'arg2' to represent the triple in the undirected KG.

TODO: verify the entries are robust, and extend parsing to some of the next ~2 mil usable English examples.

Example concept inputs:
/c/en/target
/c/en/target/n
/c/en/target/v
/c/en/target/a
/c/en/target/s
/c/en/target/r
/c/en/abandon/v/wikt/en_1
http://sw.opencyc.org/2012/05/10/concept/en/Religious_scholar
"""

import datasets
import pandas as pd
import re


data = datasets.load_dataset("conceptnet5", split="train")


def language_filter(entry):
    """TODO: Make more robust to Emojis and other nonstandard symbols. This filter is underspecified, meaning that it gets all pure english concepts, but also some more stuff that needs to be processed finer"""
    # chinese_regex = r'\/zh\/' # includes mandarin, cantonese, and other traditional/simplified texts
    english_regex = r"\/en\/"
    if re.search(english_regex, entry["arg1"]) and re.search(
        english_regex, entry["arg2"]
    ):
        return True
        # relations are all in English, even for entires with arg1 and arg2 in other languages
    else:
        return False


def parse_arg(arg):
    """Concept data is formatted: https://github.com/commonsense/conceptnet5/wiki/URI-hierarchy
    /stuff/stuff/en/arg1/optional_stuff => arg1
    """
    arg = str(arg)
    arglist = list(filter(None, arg.split("/")))
    if len(arglist) == 3:
        # have only three args
        concept = arglist[-1]
    elif len(arglist) == 4:
        # have a 4th optional argument
        concept = arglist[-2]
    elif arglist[0] == "c" and arglist[1] == "en":
        concept = arglist[2]
    elif arglist[0] == "http:":
        assert arglist[-2] == "en" and arglist[-3] == "concept", arglist
        concept = arglist[-1]
    else:
        raise Exception(f" Unexpected input {arg}, parsed as {arglist}")
    concept = str(concept)
    # replace underscore with spaces
    concept = " ".join(list(filter(None, concept.split("_"))))
    return concept


def parse_rel(rel):
    rellist = list(filter(None, rel.split("/")))
    if len(rellist) == 2:
        return rellist[-1]
    else:
        raise Exception(f" Unexpected input {rel}, parsed as {rellist}")


def english_map(entry):
    # /r/relation => relation
    new_rel = parse_rel(entry["rel"])
    # remove camelCase with space (camel Case) and lowercase (camel case)
    new_rel = re.sub("([a-z])([A-Z])", "\g<1> \g<2>", new_rel)
    new_rel = new_rel.lower()
    arg1 = parse_arg(entry["arg1"])
    # /stuff/stuff/en/arg2/optional_stuff => arg2
    arg2 = parse_arg(entry["arg2"])
    new_entry = {"arg1": arg1, "arg2": arg2, "rel": new_rel}
    return new_entry


# English data shape: (3480905, 8)
eng_data = data.filter(language_filter)
print(f"Eng data shape: {eng_data.shape}")

# to get the first million ish entries of the data
third_data = eng_data.filter(lambda example, idx: idx < 1133083, with_indices=True)

filtered_data = third_data.map(english_map)

# only interest in rel, arg1, arg2 columns
removed_cols = ["sentence", "full_rel", "extra_info", "weight", "lang"]
filtered_data = filtered_data.remove_columns(removed_cols)

df: pd.DataFrame = filtered_data.to_pandas("one_million_examples.csv")

# We need tsv format
df.to_csv("one_million_examples_four_cols.tsv", sep="\t", encoding="utf-8", index=False)



concept.startswith("/c/en/")