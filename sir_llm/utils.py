import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import os.path as osp
import ssl
import urllib.request
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="models/llama/llama-7b"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")

    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/debug",
    )

    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--recent_size", type=int, default=255)
    parser.add_argument("--enable_pos_shift", action="store_true")

    parser.add_argument("--num_eval_tokens", type=int, default=None)

    args = parser.parse_args()
    return args


def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        cache_dir="cache",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir="cache",
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def keys_dataprocessing(list_data_dict):
    final_list_data_dict = []
    for data_dict in list_data_dict:
        q= data_dict["question"]["stem"].strip()
        if "key" in data_dict["id"]:
            question=q
            choices=["OK"]
            answers=data_dict["answerKey"]
            answers_id=0
        else:
            question= q +" Choices: "
            #question= data_dict["question"]["stem"]+" Choices: "
            choices=[]
            answers=data_dict["answerKey"]
            for id,choice in enumerate(data_dict["question"]["choices"]):
                question+=" ("+choice["label"]+") "+choice["text"]
                choices.append(choice["label"])
                if choice["label"]==answers:
                    answers_id=id
        #question+=" Answer:"
        final_list_data_dict.append({"id":data_dict["id"],"question":question,"choices":choices,"answer":answers,"answer_id":answers_id})
    return final_list_data_dict   

def load_keys_jsonl(
    file_path,
):
    final_dataset=[]
    test_filepath = os.path.join(file_path, f"keys.jsonl")
    print(f"Loading data from {test_filepath} ...")
    list_data_dict = []
    with open(test_filepath, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    if "keys" in test_filepath:
        list_data_dict=keys_dataprocessing(list_data_dict)
    final_dataset+=list_data_dict
    return final_dataset


def split_string_into_tokens(string, token_length):
    return [string[i:i + token_length] for i in range(0, len(string), token_length)]


def dailydialog_dataprocessing(list_data_dict):
    final_list_data_dict = []
    for data_dict in list_data_dict:
        q= data_dict["question"]["stem"].strip()
        question= "Choose the best response for the input: "+ q +" Choices: "
        choices=[]
        answers=data_dict["answerKey"]
        for id,choice in enumerate(data_dict["question"]["choices"]):
            question+=" ("+choice["label"]+") "+choice["text"]
            choices.append(choice["label"])
            if choice["label"]==answers:
                answers_id=id
        final_list_data_dict.append({"id":data_dict["id"],"question":question,"choices":choices,"answer":answers,"answer_id":answers_id})
    return final_list_data_dict

def load_dalydialog(file_path):
    test_filepath = os.path.join(file_path, "dailydialog.jsonl")
    print(f"Loading data from {test_filepath} ...")
    list_data_dict = []
    with open(test_filepath, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    list_data_dict=dailydialog_dataprocessing(list_data_dict)
    list_data_dict=list_data_dict[0:2000] #we only use 2000 samples in the paper
    return list_data_dict


def rps_dataprocessing(list_data_dict):
    final_list_data_dict = []
    for data_dict in list_data_dict:
        question= "Play a game of Rock-Paper-Scissors. Analyze the user's previous behavior and choose one option from: rock, paper, and scissors"
        #question= data_dict["question"]["stem"]+" Choices: "
        choices=[]
        choices_letter=[]
        answers=data_dict["answerKey"]
        for id,choice in enumerate(data_dict["question"]["choices"]):
            choices.append(choice["text"])
            choices_letter.append(choice["label"])
            if choice["label"]==answers:
                answers_id=id
            if choice["label"]==data_dict["user_choice"]:
                user_choice_id=id
                user_choice=choice["text"]
        #question+=" Answer:"
        final_list_data_dict.append({"id":data_dict["id"],"question":question,"choices":choices,"choices_letter":choices_letter,"answer":answers,"answer_id":answers_id,"user_choice":user_choice,"user_choice_id":user_choice_id})
    return final_list_data_dict
    
def load_rps(file_path,domi):
    test_filepath = os.path.join(file_path, f"rock_paper_scissors_dominant_{domi}.jsonl")
    print(f"Loading data from {test_filepath} ...")
    list_data_dict = []
    with open(test_filepath, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    list_data_dict=rps_dataprocessing(list_data_dict)
    return list_data_dict