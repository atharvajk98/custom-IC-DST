import os
import json
import argparse
import copy
from collections import defaultdict
from tqdm import tqdm
from utils.helper import SpeedLimitTimer, PreviousStateRecorder, set_random_seed
from utils.typo_fix import typo_fix

import time
from gpt_turbo_completion import gpt_turbo_completion
from utils.sql import sql_pred_parse, sv_dict_to_string
from prompting import get_prompt, conversion, table_prompt
from retriever.embed_based_retriever import EmbeddingRetriever
from evaluate_metrics import evaluate
import openai

API_KEY = "OPENAI_API_KEY"
# API_KEY = "ASYNC_OPENAI_API_KEY"
# API_KEY = "SHRUTI_OPENAI_API_KEY"
# API_KEY = "ANDY_OPENAI_API_KEY"
# API_KEY = "JOEL_OPENAI_API_KEY"

openai.api_key = os.environ[API_KEY]
print(f"\nUsing {API_KEY}\n")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--seed', 
    type=int,
    default=0,
    help="Seed for reproducibility"
)
parser.add_argument(
    '--mwz_ver', 
    type=str, 
    default="2.4",
    choices=['2.1', '2.4'], help="version of MultiWOZ"
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
parser.add_argument(
    '--scratch_retriever',
    action='store_true',
    help='use off-the-shelf retriever'
)
parser.add_argument(
    '--retriever_dir', 
    type=str, 
    help="sentence transformer saved path"
)
parser.add_argument(
    '--num_examples', 
    type=int, 
    default=10,
    help="Number of examples for in-context learning"
)
parser.add_argument(
    '--train_fn', 
    type=str, 
    help="training data file (few-shot or full shot)", 
)
parser.add_argument(
    '--test_fn', 
    type=str, default='',
    help="file to evaluate on, empty means use the test set"
)
parser.add_argument(
    '--output_dir', 
    type=str, 
    default="IC-DST/expts/few-shot",
    help="directory to save running log and configs"
)
args = parser.parse_args()

print(f"\nProcess ID: {os.getpid()}\n")

# Seed the code
set_random_seed(args.seed)

print("\nArguments: ")
for key, value in vars(args).items():
    print(f"{key}: {value}")

# ----------------------------- Config -----------------------------

args.output_dir = os.path.join(args.output_dir, f"gpt_turbo_mw{args.mwz_ver}_{args.pct}p_v{args.version}")
args.output_dir = f"{args.output_dir}_scratch" if args.scratch_retriever else args.output_dir
print(f"\nStoring results at output_dir: {args.output_dir}\n")

if args.pct == 100:
    args.retriever_dir = "./src/retriever/indices/all_mpnet_base_v2_100p_v1"
    args.train_fn = f"./data/dataset/train_100p_v1.json"
else:
    args.train_fn = f"./data/dataset/train_{args.pct}p_v{args.version}.json"
print(f"\ntrain file: {args.train_fn}\n")

# create the output folder
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

# read the selection pool
with open(args.train_fn) as f:
    train_set = json.load(f)

# read the ontology and the test set
if args.mwz_ver == '2.1':
    ontology_path = "./data/mwz2.1/ontology.json"
    if args.test_fn == "":
        test_set_path = "./data/dataset/mw21_100p_test.json"
else:
    ontology_path = "./data/mwz2.4/ontology.json"
    if args.test_fn == "":
        test_set_path = "./data/dataset/mw24_100p_test.json"

# evaluate on some other file
if args.test_fn:
    test_set_path = args.test_fn

print(f"\nontology file: {ontology_path}\n")
with open(ontology_path) as f:
    ontology = json.load(f)
    
print(f"\ntest file: {test_set_path}\n")
with open(test_set_path) as f:
    test_set = json.load(f)

# load the retriever
if args.scratch_retriever:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    print(f"\nUsing retriever: {model_name}\n")
    index_file = "train_embeddings.npy"
    print("\nusing off-the-shelf retriever...\n")
    retriever = EmbeddingRetriever(
        datasets=[train_set], 
        model_path=model_name,
        search_index_filename=os.path.join(args.retriever_dir, index_file), 
        sampling_method="pre_assigned",
    )
else:
    if "original" in args.retriever_dir:
        index_file = "train_index_paraphrase.npy" if args.paraphrase else "train_index.npy"
    else:
        index_file = "train_index.npy"
    
    print("\nusing fine-tuned retriever...\n")
    retriever = EmbeddingRetriever(
        datasets=[train_set], 
        model_path=args.retriever_dir,
        search_index_filename=os.path.join(args.retriever_dir, index_file), 
        sampling_method="pre_assigned",
    )


def run(test_set, turn=-1, use_gold=False):
    # turn and use_gold are for analysis purpose
    # turn = -1 means evalute all dialogues
    # turn = 0 means evaluate single-turn dialogues
    # turn = 1 means evalute two-turn dialogues... etc.
    # when use_gold = True, the context are gold context (for analysis purpose)

    timer = SpeedLimitTimer(second_per_step=3.1)  # openai limitation 20 queries/min

    result_dict = defaultdict(list)  # use to record the accuracy

    selected_set = test_set
    # if needed, only evaluate on particular turns (analysis purpose)
    if turn >= 0:
        if not use_gold:
            raise ValueError("can only evaluate particular turn when using gold context")
        selected_set = [d for d in test_set if len(d['dialog']['usr']) == turn + 1]
    
    prediction_recorder = PreviousStateRecorder()  # state recorder

    # start experiment
    all_result = []
    n_total = 0
    n_correct = 0
    total_acc = 0
    total_f1 = 0
    start = time.time()

    for data_item in tqdm(selected_set):
        n_total += 1

        completion = ""
        if use_gold:
            prompt_text = get_prompt(data_item, examples=retriever.item_to_nearest_examples(data_item, k=args.num_examples))
        else:
            predicted_context = prediction_recorder.state_retrieval(data_item)
            modified_item = copy.deepcopy(data_item)
            modified_item['last_slot_values'] = predicted_context
            examples = retriever.item_to_nearest_examples(
                modified_item, k=args.num_examples)
            prompt_text = get_prompt(
                data_item, examples=examples, given_context=predicted_context)

        # print the retrieved examples (without the sql table)
        print(prompt_text.replace(conversion(table_prompt), ""))
        
        # record the prompt
        data_item['prompt'] = prompt_text

        # codex completion
        complete_flag = False
        parse_error_count = 0
        while not complete_flag:
            try:
                completion = gpt_turbo_completion(prompt_text)
                # convert back the sql completion result
                completion = conversion(completion, reverse=True)
            except Exception as e:
                if e.user_message.startswith("This model's maximum context length"):
                    print("prompt overlength")
                    examples = examples[1:]
                    prompt_text = get_prompt(
                        data_item, examples=examples, given_context=predicted_context)
                else:
                    # throughput too high
                    timer.sleep(10)
            else:
                try:
                    # check if CODEX is crazy
                    temp_parse = sql_pred_parse(completion)
                except:
                    parse_error_count += 1
                    if parse_error_count >= 5:
                        complete_flag = True
                else:
                    complete_flag = True
            # limit query speed
            timer.step()
            
        # aggregate the prediction and the history states
        predicted_slot_values = {}
        try:
            predicted_slot_values = sql_pred_parse(completion)  # a dictionary
        except:
            print("the output is not a valid SQL query")
            data_item['not_valid'] = 1

        predicted_slot_values = typo_fix(predicted_slot_values, ontology=ontology, version=args.mwz_ver)
        context_slot_values = data_item['last_slot_values']  # a dictionary

        # merge context and prediction
        if use_gold:
            all_slot_values = context_slot_values.copy()
        else:
            all_slot_values = prediction_recorder.state_retrieval(
                data_item).copy()

        for s, v in predicted_slot_values.items():

            if s in all_slot_values and v == "[DELETE]":
                del all_slot_values[s]
            elif v != "[DELETE]":
                all_slot_values[s] = v

        # some slots may contain multiple values
        all_slot_values = {k: v.split('|')[0]
                           for k, v in all_slot_values.items()}

        # record current turn prediction
        prediction_recorder.add_state(data_item, all_slot_values)

        # record the predictions
        data_item['pred'] = all_slot_values
        data_item['ontology_path'] = ontology_path
        data_item['completion'] = completion
        all_result.append(data_item)

        # print the result
        print(completion)
        print(
            f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
        print(
            f"pred turn change: {sv_dict_to_string(predicted_slot_values, sep='-')}")
        print(
            f"gold turn change: {sv_dict_to_string(data_item['turn_slot_values'], sep='-')}")
        print(f"pred states: {sv_dict_to_string(all_slot_values, sep='-')}")
        print(
            f"gold states: {sv_dict_to_string(data_item['slot_values'], sep='-')}")

        this_jga, this_acc, this_f1 = evaluate(
            all_slot_values, data_item['slot_values'])
        total_acc += this_acc
        total_f1 += this_f1

        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']].append(1)
            print("\n=====================correct!=======================")
        else:
            result_dict[data_item['turn_id']].append(0)
            print("\n=====================wrong!=======================")

        print("\n")

    end = time.time()
    print(f"\nTotal time taken = {end - start}\n")
    
    print(f"correct {n_correct}/{n_total}  =  {n_correct / n_total}")
    print(f"Slot Acc {total_acc/n_total}")
    print(f"Joint F1 {total_f1/n_total}")
    print()

    # calculate the accuracy of each turn
    for k, v in result_dict.items():
        print(f"accuracy of turn {k} is {sum(v)}/{len(v)} = {sum(v) / len(v)}")

    return all_result



if __name__ == "__main__":

    all_results = run(test_set)

    with open(os.path.join(args.output_dir, "running_log.json"), 'w') as f:
        json.dump(all_results, f, indent=4)
