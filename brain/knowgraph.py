# coding: utf-8
"""
KnowledgeGraph
"""
import re
import numpy as np
import os


if __name__ == "__main__":
    class config:
        FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

        KGS = {
            'ConceptNet': 'preprocess/conceptnet5/one_million_examples.tsv',
        }

        MAX_ENTITIES = 2

        # Special token words.
        PAD_TOKEN = '[PAD]'
        UNK_TOKEN = '[UNK]'
        CLS_TOKEN = '[CLS]'
        SEP_TOKEN = '[SEP]'
        MASK_TOKEN = '[MASK]'
        ENT_TOKEN = '[ENT]'
        SUB_TOKEN = '[SUB]'
        PRE_TOKEN = '[PRE]'
        OBJ_TOKEN = '[OBJ]'

        NEVER_SPLIT_TAG = [
            PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN,
            ENT_TOKEN, SUB_TOKEN, PRE_TOKEN, OBJ_TOKEN
        ]
else:
    import brain.config as config

class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, spo_files, predicate=False):
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        self.text = " ".join(self.segment_vocab)
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")    
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate:
                        value = pred + " " + obje
                    else:
                        value = obje
                    if subj in lookup_table:
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        # remove column labels
        if lookup_table.get('arg1'):
            del lookup_table['arg1']
        return lookup_table

    def tokenize(self, text:str):
        """Tokenizer. Want to make tokens NEVER_SPLIT_TAG be parsed as a single token. Other words and numbers should be parsed
        according to vocab.txt and appearances in the ConceptNet KG."""
        expand_punc_regex = r"[a-zA-Z']+|[.,?!]|(\[CLS\])|(\[SEP\])|(\[PAD\])|(\[UNK\])|(\[MASK\])|(\[ENT\])|(\[SUB\])|(\[PRE\])|(\[OBJ\])"
        pattern = re.compile(expand_punc_regex)
        return (match.group(0) for match in pattern.finditer(text))

    def entity_tokenize(self, text:str):
        """Returns an iterable (list or generator) of the text matched"""
        no_ws = text.split()
        res = []
        for word in no_ws:
            if ":" in word:
                w = word.split(":")[-1]
                res.append(w)
            else:
                res.append(word)
        return res

    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128, trial=0):
        """
        Args:
            sent_batch (list): list of sentences, always 1
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        if len(sent_batch) > 1: 
            print("actual batch found, expected only 1 sent", sent_batch)
            raise Exception
        # TODO: not grouping words by semantic unit
            # TODO: NER package, spacy, or look ahead in the sent_batch and use the longest matching subj in the KG
        split_sent = self.tokenize(sent_batch[0])
        # create tree
        sent_tree = []  # tuple of word, words from knowledge graph
        pos_idx_tree = []
        abs_idx_tree = []
        pos_idx = -1
        abs_idx = -1
        abs_idx_src = []
        split_sent = self.tokenize(sent_batch[0])
        for token in split_sent:
            #TODO: add filter step before truncate to max_entities len
            filt_list = list(self.lookup_table.get(token, []))
            entities = filt_list[:max_entities]
            sent_tree.append((token, entities))

            token_pos_idx = [pos_idx +1]
            token_abs_idx = [abs_idx+1]
            abs_idx = token_abs_idx[-1]
            entities_pos_idx = []
            entities_abs_idx = []
            for ent in entities:
                list_words_in_ent = list(self.entity_tokenize(ent))
                ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(list_words_in_ent)+1)]
                entities_pos_idx.append(ent_pos_idx)
                ent_abs_idx = [abs_idx + i for i in range(1, len(list_words_in_ent)+1)]
                abs_idx = ent_abs_idx[-1]
                entities_abs_idx.append(ent_abs_idx)
            pos_idx_tree.append((token_pos_idx, entities_pos_idx))
            pos_idx = token_pos_idx[-1]
            abs_idx_tree.append((token_abs_idx, entities_abs_idx))
            abs_idx_src += token_abs_idx
        know_sent = []
        pos = []
        seg = []
        for i in range(len(sent_tree)):
            word = sent_tree[i][0]
            if word in self.special_tags:
                know_sent += [word]
            else:
                add_word = [word]
                know_sent += add_word
                if len(add_word) > 1:
                    raise Exception("Found anomaly")
            seg += [0]
            pos += pos_idx_tree[i][0]
            for j in range(len(sent_tree[i][1])):
                add_word_list = sent_tree[i][1][j].split(' ')
                know_sent += add_word_list
                seg += [1] * len(add_word_list)
                pos += list(pos_idx_tree[i][1][j])

        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        token_num = len(know_sent)

        # Calculate visible matrix
        visible_matrix = np.zeros((token_num, token_num))
        for item in abs_idx_tree:
            src_ids = item[0]
            for id in src_ids:
                visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                visible_matrix[id, visible_abs_idx] = 1
            for ent in item[1]:
                for id in ent:
                    visible_abs_idx = ent + src_ids
                    visible_matrix[id, visible_abs_idx] = 1

        src_length = len(know_sent)
        if len(know_sent) < max_length:
            pad_num = max_length - src_length
            know_sent += [config.PAD_TOKEN] * pad_num
            seg += [0] * pad_num
            pos += [max_length - 1] * pad_num
            visible_matrix = np.pad(
                visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
        else:
            know_sent = know_sent[:max_length]
            seg = seg[:max_length]
            pos = pos[:max_length]
            visible_matrix = visible_matrix[:max_length, :max_length]

        know_sent_batch.append(know_sent)
        position_batch.append(pos)
        visible_matrix_batch.append(visible_matrix)
        seg_batch.append(seg)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch


if __name__ == "__main__":
    s = ['[CLS] Rocks are classified as igneous, metamorphic, or sedimentary according to [SEP] their color [SEP]']
    sent_batch = " ".join(s)
    trials = 1
    for j in range(trials):
        k = KnowledgeGraph(['ConceptNet'], predicate=True)
        count = 0
        try:
            kn_s,p,v,s = k.add_knowledge_with_vm([sent_batch], trial=j)
            count +=1 
        except:
            print(f'Which Sentence: {count}')
        
    # print('k\n',k, 'p\n', p, 'v\n', v, 's\n', s)
    