import os

FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

KGS = {
    'HowNet': os.path.join(FILE_DIR_PATH, 'kgs/HowNet.spo'),
    'CnDbpedia': os.path.join(FILE_DIR_PATH, 'kgs/CnDbpedia.spo'),
    'chineseconceptnet': os.path.join(FILE_DIR_PATH, 'kgs/chineseconceptnet.spo'),
    'sent_dic':os.path.join(FILE_DIR_PATH, 'kgs/sent.spo'),
}

MAX_ENTITIES = 2

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
