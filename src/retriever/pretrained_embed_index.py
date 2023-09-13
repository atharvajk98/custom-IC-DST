import argparse
import os
import json
import jsonlines
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModel
)
from retriever_utils import set_random_seed

# ------ Configuration ends here ----------------

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--seed', 
    type=int, 
    default=0, 
    help='Seed used to reproduce results'
)
parser.add_argument(
    '--mwz_ver', 
    type=str, 
    default="2.4",
    choices=['2.1', '2.4'], 
    help="version of MultiWOZ"
)
parser.add_argument(
    '--pct', 
    type=int, 
    default=5,
    help='Pct split of the synthetic data'
)
parser.add_argument(
    '--version', 
    type=int, 
    default=1,
    choices=[1, 2, 3],
    help='different versions of the data'
)
args = parser.parse_args()

set_random_seed(args.seed)

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
SAVE_NAME = f'all_mpnet_base_v2_mw{args.mwz_ver}_{args.pct}p_v{args.version}'

train_file = f"./data/dataset/train_{args.pct}p_v{args.version}.json"
dev_file = f"./data/dataset/mw{args.mwz_ver.replace('.', '')}_100p_dev.json"
test_file = f"./data/dataset/mw{args.mwz_ver.replace('.', '')}_100p_test.json"

print(f"\ntrain_file: {train_file}")
print(f"\ndev_file: {dev_file}")
print(f"\ntest_file: {test_file}")

# path to save indexes and results
save_path = f"./src/retriever/indices/{SAVE_NAME}"
os.makedirs(save_path, exist_ok = True) 

CLS_Flag = False

def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.cuda()


# function for embedding one string
def embed_single_sentence(sentence, cls=CLS_Flag):
    
    # Sentences we want sentence embeddings for
    sentences = [sentence]

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True,
                          truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids'].cuda()
    attention_mask = encoded_input['attention_mask'].cuda()

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)

    # Perform pooling
    sentence_embeddings = None

    if cls:
        sentence_embeddings = model_output[0][:,0,:]
    else:
        sentence_embeddings = mean_pooling(model_output, attention_mask)

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


def read_MW_dataset(mw_json_fn):

    # only care domain in test
    DOMAINS = ['hotel', 'restaurant', 'attraction', 'taxi', 'train']

    with open(mw_json_fn, 'r') as f:
        data = json.load(f)

    dial_dict = {}

    for turn in data:
        # filter the domains that not belongs to the test domain
        if not set(turn["domains"]).issubset(set(DOMAINS)):
            continue

        # update dialogue history
        sys_utt = turn["dialog"]['sys'][-1]
        usr_utt = turn["dialog"]['usr'][-1]

        if sys_utt == 'none':
            sys_utt = ''
        if usr_utt == 'none':
            usr_utt = ''

        history = f"[system] {sys_utt} [user] {usr_utt}"

        # store the history in dictionary
        name = f"{turn['ID']}_turn_{turn['turn_id']}"
        dial_dict[name] = history

    return dial_dict


mw_train = read_MW_dataset(train_file)
mw_dev = read_MW_dataset(dev_file)
mw_test = read_MW_dataset(test_file)
print("Finish reading data")

def store_embed(input_dataset, output_filename, forward_fn):
    outputs = {}
    with torch.no_grad():
        for k, v in tqdm(input_dataset.items()):
            outputs[k] = forward_fn(v).detach().cpu().numpy()
    np.save(output_filename, outputs)
    return

# store the embeddings
store_embed(mw_train, f"{save_path}/train_embeddings.npy",
            embed_single_sentence)
store_embed(mw_dev, f"{save_path}/dev_embeddings.npy",
            embed_single_sentence)
store_embed(mw_test, f"{save_path}/test_embeddings.npy",
            embed_single_sentence)
print("Finish Embedding data")