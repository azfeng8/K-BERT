# -*- encoding:utf-8 -*-
"""
  This script provides an k-BERT exmaple for classification.
"""
# import pdb
import sys
import torch
import random
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_model
from uer.utils.optimizers import  BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from brain import KnowledgeGraph
from multiprocessing import Pool
import numpy as np


# Supress warnings
import warnings
warnings.filterwarnings("ignore")


class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.use_vm = False if args.no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        preds = self.softmax(logits.view(-1, self.labels_num))
        loss = self.criterion(preds, label.view(-1))

        return loss, logits

def add_knowledge_worker(params, verbose=True):

    p_id, sentences, columns, kg, vocab, args = params

    sentences_num = len(sentences)
    dataset = []
    for line_id, line in enumerate(sentences):
        if line_id % 10000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        line = line.strip().split('\t')
        if len(line) == 4:
            # Original NLPCC-DBQA does similar to this
            # ID (str), Question (str), answer (str), label (0 or 1)
            qid=int(line_id)
            label = int(line[3])
            question = line[1]
            answer = line[2]
            text = CLS_TOKEN + " " +  question +  " " + SEP_TOKEN +  " " + answer +  " " + SEP_TOKEN
            try:
                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
            except IndexError as e:
                print(e)
                print("Dataset sentence error:", [text])
            tokens = tokens[0]
            pos = pos[0]
            vm = vm[0].astype("bool")

            token_ids = [vocab.get(t) for t in tokens]
            mask = []
            seg_tag = 1
            for t in tokens:
                if t == PAD_TOKEN:
                    mask.append(0)
                else:
                    mask.append(seg_tag)
                if t == SEP_TOKEN:
                    seg_tag += 1
            
            dataset.append((token_ids, label, mask, pos, vm, qid))
        else:
            if verbose: print("got unexpected input, ignoring", line)
    return dataset

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    # parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")
    parser.add_argument("--lang", choices=["zh", "en"], help="Language to use for tokenizer")


    # Options for saving and loading model to resume training
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to save model for training.")
    parser.add_argument("--save_model_path", default=None, type=str,
                        help="Path to load model for training.")
    parser.add_argument("--test_only", action="store_true",
                        help="Only run code to evaluate model")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set) 

    # Hardcode number of labels
    args.labels_num = 2

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path:
        print(f"Initializing model from pretrained model from {args.pretrained_model_path}")
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)  
        # Build classification model.
        model = BertClassifier(args, model)
    
    if args.load_model_path:
        print(f"Loading saved model from {args.load_model_path}")
        model = torch.load(args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            vms_batch = vms[i*batch_size: (i+1)*batch_size]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            vms_batch = vms[instances_num//batch_size*batch_size:]

            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]

    if args.lang == "zh":
        raise Exception("Deprecated, removed Chinese version") 
    else:
        kg = KnowledgeGraph(spo_files=spo_files, predicate=True)

    def read_dataset(path, workers_num=1):

        print("Loading sentences from {}".format(path))
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
        sentence_num = len(sentences)

        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
        if workers_num > 1:
            params = []
            sentence_per_block = int(sentence_num / workers_num) + 1
            for i in range(workers_num):
                params.append((i, sentences[i*sentence_per_block: (i+1)*sentence_per_block], columns, kg, vocab, args))
            pool = Pool(workers_num)
            res = pool.map(add_knowledge_worker, params)
            pool.close()
            pool.join()
            dataset = [sample for block in res for sample in block]
        else:
            params = (0, sentences, columns, kg, vocab, args)
            dataset = add_knowledge_worker(params)

        return dataset

    # Evaluation function.
    def evaluate(args, is_test, metrics='Acc'):
        losses = []
        if is_test:
            dataset = read_dataset(args.test_path, workers_num=args.workers_num)
        else:
            dataset = read_dataset(args.dev_path, workers_num=args.workers_num)

        #TODO: this is faster in one for loop
        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([example[3] for example in dataset])
        vms = [example[4] for example in dataset]

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()
        
        # pdb.set_trace()
        for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

            # vms_batch = vms_batch.long()
            vms_batch = torch.LongTensor(vms_batch)

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            with torch.no_grad():
                try:
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                    losses.append(loss.item())
                except:
                    print(input_ids_batch)
                    print(input_ids_batch.size())
                    print(vms_batch)
                    print(vms_batch.size())
                    print("loss", loss)

            logits = nn.Softmax(dim=1)(logits)
            pred = torch.argmax(logits, dim=1)
            gold = label_ids_batch
            # pdb.set_trace()
            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred == gold).item()
    
        plt.figure()
        plt.title("Evaluation: loss over time")
        plt.xlabel("Batch num")
        plt.ylabel("loss")
        plt.plot(np.arange(len(losses)), np.array(losses))
        plt.savefig('eval_loss.png')
        plt.show()

        print("confusion matrix:")
        print(confusion)
        print("Report precision, recall, and f1:")
        
        for i in range(confusion.size()[0]):
            pred_1 = confusion[i,:].sum().item()
            if np.isclose(pred_1, 0):
                print("Warning: All predictions are 0, can't comput precision score")
                p = np.nan
            else:
                p = confusion[i,i].item()/pred_1
            r = confusion[i,i].item()/confusion[:,i].sum().item()
            f1 = 2*p*r / (p+r)
            if i == 1:
                label_1_f1 = f1
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
        print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(dataset), correct, len(dataset)))
        if metrics == 'Acc':
            return correct/len(dataset)
        elif metrics == 'f1':
            return label_1_f1
        else:
            return correct/len(dataset)
    # Training phase.

    if args.test_only:
        print("Evaluating on the test dataset.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, True)
        return

    print("Start training.")
    trainset = read_dataset(args.train_path, workers_num=args.workers_num)
    print("Shuffling dataset")
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    print("Trans data to tensor.")
    print("input_ids")
    input_ids = torch.LongTensor([example[0] for example in trainset])
    print("label_ids")
    label_ids = torch.LongTensor([example[1] for example in trainset])
    print("mask_ids")
    mask_ids = torch.LongTensor([example[2] for example in trainset])
    print("pos_ids")
    pos_ids = torch.LongTensor([example[3] for example in trainset])
    print("vms")
    vms = [example[4] for example in trainset]

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    result = 0.0
    best_result = 0.0
    
    train_loss = []
    for epoch in range(1, args.epochs_num+1):
        model.train()
        epoch_loss = []
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):
            model.zero_grad()

            vms_batch = torch.LongTensor(np.array(vms_batch))

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos=pos_ids_batch, vm=vms_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            epoch_loss.append(loss.item())
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                sys.stdout.flush()
                total_loss = 0.
            loss.backward()
            optimizer.step()

        train_loss.append(epoch_loss)

        print("Start evaluation on dev dataset.")
        result = evaluate(args, False)
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)
        else:
            continue
        
        # Skip evaluation on test during training (there's a final test eval at the end)
        # print("Start evaluation on test dataset.")
        # evaluate(args, True)

    # Save model
    if args.save_model_path:
        print(f"Saving model to {args.save_model_path}")
        torch.save(model, args.save_model_path)
        print(f"Saving training losses to {args.save_model_path[:-3]}_losses.pt")
        torch.save(train_loss, f"{args.save_model_path[:-3]}_losses.pt")
    
    # plt.figure()
    # plt.title("Train loss over time (all epochs and batches)")
    # plt.plot(np.arange((len(train_loss))), train_loss)
    # plt.savefig('train_loss.png')
    # plt.show()

    # Evaluation phase.
    print("Final evaluation on the test dataset.")

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))
    evaluate(args, True)

# def debug(path):
#     model = build_model(path)
#     evaluate(get_args())

if __name__ == "__main__":
    main()