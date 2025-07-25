import os
import copy
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from transformers import BertModel, AdamW
from constants import *
from vocab import Vocab

from models.acegcn_model import ACEGCNClassifier
from brain import KnowledgeGraph

from syntax import build_multi_sentence_mask_matrices

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Instructor:

    def __init__(self, opt):
        self.opt = opt

        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)

        self.columns = {}
        with open(opt.dataset_file['train'], mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        self.columns[column_name] = i
                    continue

        self.vocab = Vocab()
        self.vocab.load(opt.vocab_path)

        if opt.kg_name == 'none':
            spo_files = []
        else:
            spo_files = [opt.kg_name]
        self.kg = KnowledgeGraph(spo_files=spo_files, predicate=True)

        trainset = self.read_dataset(opt.dataset_file['train'],workers_num=opt.workers_num)
        random.shuffle(trainset)

        print("Trans data to tensor.")
        self.input_ids_train = torch.LongTensor([example[0] for example in trainset])
        self.label_ids_train = torch.LongTensor([example[1] for example in trainset])
        self.mask_ids_train = torch.LongTensor([example[2] for example in trainset])
        self.pos_ids_train = torch.LongTensor([example[3] for example in trainset])
        self.vms_train = [example[4] for example in trainset]
        self.src_mask_train = torch.LongTensor([example[5] for example in trainset])
        self.sentence_train = [example[6] for example in trainset][0]
        self.tokens_train = [example[7] for example in trainset]

        devset = self.read_dataset(opt.dataset_file['dev'], workers_num=opt.workers_num)

        print("Trans data to tensor.")
        self.input_ids_dev = torch.LongTensor([example[0] for example in devset])
        self.label_ids_dev = torch.LongTensor([example[1] for example in devset])
        self.mask_ids_dev = torch.LongTensor([example[2] for example in devset])
        self.pos_ids_dev = torch.LongTensor([example[3] for example in devset])
        self.vms_dev = [example[4] for example in devset]
        self.src_mask_dev = torch.LongTensor([example[5] for example in devset])
        self.sentence_dev = [example[6] for example in devset][0]
        self.tokens_dev = [example[7] for example in devset]

        testset = self.read_dataset(opt.dataset_file['test'],workers_num=opt.workers_num)

        print("Trans data to tensor.")
        self.input_ids_test = torch.LongTensor([example[0] for example in testset])
        self.label_ids_test = torch.LongTensor([example[1] for example in testset])
        self.mask_ids_test = torch.LongTensor([example[2] for example in testset])
        self.pos_ids_test = torch.LongTensor([example[3] for example in testset])
        self.vms_test = [example[4] for example in testset]
        self.src_mask_test = torch.LongTensor([example[5] for example in testset])
        self.sentence_test = [example[6] for example in testset][0]
        self.tokens_test = [example[7] for example in testset]


        if opt.device.type == 'cuda':
            print('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
    def add_knowledge_worker(self, params):

        p_id, sentences, columns, kg, vocab, opt = params
        dataset = []

        for line_id, line in enumerate(sentences):

            line = line.strip().split('\t')
            label = int(line[columns["label"]])
            text = CLS_TOKEN + line[columns["text_a"]]

            tokens, pos, vm, _ = kg.add_knowledge_with_vm(
                [text], add_pad=True, max_length=opt.seq_length
            )
            tokens = tokens[0]
            pos = pos[0]
            vm = vm[0].astype("bool")

            valid_length = len([t for t in tokens if t != PAD_TOKEN])
            src_mask = [1] * valid_length + [0] * (opt.seq_length - valid_length)

            token_ids = [vocab.get(t) for t in tokens]
            mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

            dataset.append((token_ids, label, mask, pos, vm, src_mask, sentences, tokens))

        return dataset

    def read_dataset(self, path, workers_num=1):
        print("Loading sentences from {}".format(path))
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
        sentence_num = len(sentences)
        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(
            sentence_num, workers_num))

        params = (0, sentences, self.columns, self.kg, self.vocab, self.opt)
        dataset = self.add_knowledge_worker(params)

        return dataset

    def batch_loader(self, batch_size, input_ids, label_ids, mask_ids, pos_ids, vms, src_mask, sentence, tokens):
            instances_num = input_ids.size(0)
            total_batches = instances_num // batch_size
            remainder = instances_num % batch_size

            for i in range(total_batches):
                start = i * batch_size
                end = (i + 1) * batch_size

                input_ids_batch = input_ids[start:end, :]
                label_ids_batch = label_ids[start:end]
                mask_ids_batch = mask_ids[start:end, :]
                pos_ids_batch = pos_ids[start:end, :]
                vms_batch = vms[start:end]
                src_mask_batch = src_mask[start:end, :]
                sentence_batch = sentence[start:end]
                tokens_batch = tokens[start:end]

                yield (input_ids_batch, label_ids_batch, mask_ids_batch,
                       pos_ids_batch, vms_batch, src_mask_batch, sentence_batch, tokens_batch)

            if remainder > 0:
                start = total_batches * batch_size
                input_ids_batch = input_ids[start:, :]
                label_ids_batch = label_ids[start:]
                mask_ids_batch = mask_ids[start:, :]
                pos_ids_batch = pos_ids[start:, :]
                vms_batch = vms[start:]
                src_mask_batch = src_mask[start:, :]
                sentence_batch = sentence[start:]
                tokens_batch = tokens[start:]

                yield (input_ids_batch, label_ids_batch, mask_ids_batch,
                       pos_ids_batch, vms_batch, src_mask_batch, sentence_batch, tokens_batch)


    def get_bert_optimizer(self, model):

        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]

        if self.opt.diff_lr:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer

    
    def _train(self, criterion, optimizer, max_test_acc_overall=0):


        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        for epoch in range(self.opt.num_epoch):

            n_correct, n_total = 0, 0
            for i_batch, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, src_mask_batch, sentence_batch, tokens_batch) in enumerate(self.batch_loader(self.opt.batch_size, self.input_ids_train, self.label_ids_train, self.mask_ids_train, self.pos_ids_train, self.vms_train, self.src_mask_train, self.sentence_train, self.tokens_train)):
                global_step += 1

                self.model.train()
                optimizer.zero_grad()

                vms_batch = torch.tensor(np.array(vms_batch).astype(np.int64), dtype=torch.long)

                input_ids_batch = input_ids_batch.to(self.opt.device)
                label_ids_batch = label_ids_batch.to(self.opt.device)
                mask_ids_batch = mask_ids_batch.to(self.opt.device)
                pos_ids_batch = pos_ids_batch.to(self.opt.device)
                vms_batch = vms_batch.to(self.opt.device)
                src_mask_batch = src_mask_batch.to(self.opt.device)

                syntax_matrix_batch, _ = build_multi_sentence_mask_matrices(sentence_batch, tokens_batch, self.opt.attention_heads, self.opt.seq_length)
                syntax_matrix_batch = torch.FloatTensor(syntax_matrix_batch).to(self.opt.device)

                outputs, penal = self.model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, src_mask_batch, syntax_matrix_batch)
                targets = label_ids_batch

                if self.opt.losstype is not None:
                    loss = criterion(outputs, targets) + penal
                else:
                    loss = criterion(outputs, targets)

                print(global_step)
                print(loss)

                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:  

                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate(False,False)
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./state_dict'):
                                os.mkdir('./state_dict')
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name, self.opt.dataset, test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                    if f1 > max_f1:
                        max_f1 = f1
        return max_test_acc, max_f1, model_path
    
    def _evaluate(self, show_results=False, is_Test=False):

        self.model.eval() 
        n_test_correct, n_test_total = 0, 0 
        targets_all, outputs_all = None, None 
        with ((torch.no_grad())):
            if(is_Test==True):
                batch_size= self.opt.batch_size
                input_ids_test= self.input_ids_test
                label_ids_test= self.label_ids_test
                mask_ids_test= self.mask_ids_test
                pos_ids_test= self.pos_ids_test
                vms_test= self.vms_test
                src_mask_test= self.src_mask_test
                sentence_test= self.sentence_test
                tokens_test = self.tokens_test
            elif(is_Test==False):
                batch_size = self.opt.batch_size
                input_ids_test = self.input_ids_dev
                label_ids_test = self.label_ids_dev
                mask_ids_test = self.mask_ids_dev
                pos_ids_test = self.pos_ids_dev
                vms_test = self.vms_dev
                src_mask_test = self.src_mask_dev
                sentence_test = self.sentence_dev
                tokens_test = self.tokens_dev

            for i_batch, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, src_mask_batch, sentence_batch, tokens_batch) in enumerate(self.batch_loader(batch_size, input_ids_test, label_ids_test, mask_ids_test, pos_ids_test, vms_test, src_mask_test, sentence_test, tokens_test)):

                vms_batch = torch.tensor(np.array(vms_batch).astype(np.int64), dtype=torch.long)

                input_ids_batch = input_ids_batch.to(self.opt.device)
                label_ids_batch = label_ids_batch.to(self.opt.device)
                mask_ids_batch = mask_ids_batch.to(self.opt.device)
                pos_ids_batch = pos_ids_batch.to(self.opt.device)
                vms_batch = vms_batch.to(self.opt.device)
                src_mask_batch = src_mask_batch.to(self.opt.device)

                syntax_matrix_batch, _ = build_multi_sentence_mask_matrices(sentence_batch, tokens_batch, self.opt.attention_heads,self.opt.seq_length)
                syntax_matrix_batch = torch.FloatTensor(syntax_matrix_batch).to(self.opt.device)

                targets = label_ids_batch
                outputs, penal = self.model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, src_mask_batch, syntax_matrix_batch)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1], average='binary')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1
        return test_acc, f1

    def _test(self):

        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True, is_Test=True)
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
        
    
    def run(self):

        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_bert_optimizer(self.model)

        max_test_acc_overall = 0 
        max_f1_overall = 0 


        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
        print('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))

        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)

        torch.save(self.best_model.state_dict(), model_path)
        print('>> saved: {}'.format(model_path))

        print('#' * 60)
        print('max_test_acc_overall:{}'.format(max_test_acc_overall))
        print('max_f1_overall:{}'.format(max_f1_overall))
        self._test()

def main():


    model_classes = {
        'acegcn': ACEGCNClassifier,
    }
    
    dataset_files = {
        'English': {
            'train': './datasets/mooc_eng/train.tsv',
            'test': './datasets/mooc_eng/test.tsv',
            'dev': './datasets/mooc_eng/dev.tsv'
        }
        'Computer': {
            'train': './datasets/mooc/train.tsv',
            'test': './datasets/mooc/test.tsv',
            'dev': './datasets/mooc/dev.tsv'
        }
        'Mix': {
            'train': './datasets/mooc_mix/train.tsv',
            'test': './datasets/mooc_mix/test.tsv',
            'dev': './datasets/mooc_mix/dev.tsv'
        }
    }

    input_colses = {
        'acegcn': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'src_mask', 'aspect_mask','short_mask']
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad, 
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax, 
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD, 
    }


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='acegcn', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='English', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument("--workers_num", type=int, default=1,help="number of process for loading dataset ")
    parser.add_argument("--kg_name", default='HowNet',required=False, help="KG name or path")
    parser.add_argument("--seq_length", type=int, default=128,help="Sequence length.")
    parser.add_argument("--vocab_path", default="./modelbert/vocab.txt", type=str,help="Path of the vocabulary file.")

    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=125, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=50, help='GCN mem dim.')
    parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
    parser.add_argument('--polarities_dim', default=2, type=int, help='Num of polarities')

    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0, help='GCN layer dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
    parser.add_argument('--loop', default=True)
    
    parser.add_argument('--attention_heads', default=5, type=int, help='number of multi-attention heads')
    parser.add_argument('--max_length', default=85, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--vocab_dir', type=str, default='./dataset/Laptops_corenlp')
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--parseadj', default=False, action='store_true', help='dependency probability')
    parser.add_argument('--parsehead', default=False, action='store_true', help='dependency tree')
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--losstype', default=None, type=str, help="['doubleloss', 'orthogonalloss', 'differentiatedloss']")
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--beta', default=0.25, type=float)

    parser.add_argument('--pretrained_bert_name', default='./modelbert/', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=2e-5, type=float)

    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    print("choice cuda:{}".format(opt.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)

    setup_seed(opt.seed)

    ins = Instructor(opt)
    ins.run()

if __name__ == '__main__':
    main()
