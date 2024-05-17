import warnings
 
warnings.filterwarnings("ignore")


import argparse
import json
import os


from sir_llm.utils import load, load_dalydialog,load_rps,load_keys_jsonl
from sir_llm.enable_streaming_llm import enable_streaming_llm
from sir_llm.eval_utils import Evaluator


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    evaluator= Evaluator(model, tokenizer, args)
    if "dailydialog" in args.data_root:
        prompts = load_dalydialog(args.data_root)
        if args.enable_streaming:
            if args.enable_token_entropy:
                kv_cache = enable_streaming_llm(
                    model, start_size=args.start_size, 
                    recent_size=args.recent_size,
                    token_entropy_size=args.token_entropy_size,
                )
            else:
                kv_cache = enable_streaming_llm(
                    model, start_size=args.start_size, 
                    recent_size=args.recent_size,
                    token_entropy_size=0,
                )
        else:
            kv_cache = None
        if args.enable_token_entropy:
            if args.if_w_turns:
                #start a new past key values for each turn
                acc,total_generation=evaluator.streaming_inference_token_entropy_w_turns(
                    prompts,
                    kv_cache,
                )
            else:
                #make sure to pass if_w_turns in dailydialog dataset
                raise ValueError("if_w_turns should be set to True for dailydialog dataset")
        elif args.few_shot>=0:
            acc,total_generation=evaluator.inference_few_shot(
                prompts,
                few_shot=args.few_shot,
                gold_label=args.few_shot_gold_label,
            )
        else:
            if args.if_w_turns:
                acc,total_generation=evaluator.streaming_inference_w_turns(
                    prompts,
                    kv_cache,
                )
            else:
                #make sure to pass if_w_turns in dailydialog dataset
                raise ValueError("if_w_turns should be set to True for dailydialog dataset") 
        with open(os.path.join(args.output_dir, f"dailydialog.jsonl"), "w") as f:
            args_dict = vars(args)
            args_dict["accuracy"]=acc
            args_dict["total_generation"]=total_generation
            json.dump(args_dict, f, indent=4)
    elif "grocery" in args.data_root:
        #assert False
        prompts = load_keys_jsonl(args.data_root)

        if args.enable_streaming:
            if args.enable_token_entropy:
                kv_cache = enable_streaming_llm(
                    model, start_size=args.start_size, 
                    recent_size=args.recent_size,
                    token_entropy_size=args.token_entropy_size,
                )
            else:
                kv_cache = enable_streaming_llm(
                    model, start_size=args.start_size, 
                    recent_size=args.recent_size,
                    token_entropy_size=0,
                )
        else:
            kv_cache = None
        if args.enable_token_entropy:
            acc,acc_key,total_generation,total_arc_num,total_key_num =evaluator.streaming_inference_token_entropy_for_keys(
                prompts,
                kv_cache,
            )
        elif args.few_shot>=0:
            #assert False, "few shot not supported for keys"
            acc,acc_key,total_generation,total_arc_num,total_key_num=evaluator.inference_few_shot_for_keys(
                prompts,
                few_shot=args.few_shot,
                gold_label=args.few_shot_gold_label,
            )
        else:
            acc,acc_key,total_generation,total_arc_num,total_key_num=evaluator.streaming_inference_for_keys(
                prompts,
                kv_cache,
            )
        with open(os.path.join(args.output_dir, f"key.jsonl"), "w") as f:
            args_dict = vars(args)
            args_dict["accuracy"]=acc
            args_dict["accuracy_key"]=acc_key
            args_dict["total_arc_num"]=total_arc_num
            args_dict["total_key_num"]=total_key_num
            args_dict["total_generation"]=total_generation
            json.dump(args_dict, f, indent=4)

    elif "rock_paper_scissors" in args.data_root:
        for domi in ["rock","paper","scissors"]:
            prompts = load_rps(args.data_root,domi=domi)
            prompts=prompts[:2000] #only play for 2000 rounds
            if args.enable_streaming:
                if args.enable_token_entropy:
                    kv_cache = enable_streaming_llm(
                        model, start_size=args.start_size, 
                        recent_size=args.recent_size,
                        token_entropy_size=args.token_entropy_size,
                    )
                else:
                    kv_cache = enable_streaming_llm(
                        model, start_size=args.start_size, 
                        recent_size=args.recent_size,token_entropy_size=0,
                    )
            else:
                kv_cache = None
            if args.enable_token_entropy:
                res,total_generation=evaluator.streaming_inference_token_entropy_for_rps(
                    prompts,
                    kv_cache,
                )
            elif args.few_shot>=0:
                raise NotImplementedError
            else:
                res,total_generation=evaluator.streaming_inference_for_rps(
                    prompts,
                    kv_cache,
                )
            with open(os.path.join(args.output_dir, f"rock_paper_scissors_dominant_{domi}.jsonl"), "w") as f:
                args_dict = vars(args)
                args_dict["win_rate"]=res["win_rate"]
                args_dict["tie_rate"]=res["tie_rate"]
                args_dict["lose_rate"]=res["lose_rate"]
                args_dict["exact_match_rate"]=res["exact_match_rate"]
                args_dict["total_generation"]=total_generation
                json.dump(args_dict, f, indent=4)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-7b-v1.5"
    )
    parser.add_argument("--data_root", type=str, default="data/ARC")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--enable_token_entropy", action="store_true")
    parser.add_argument("--few_shot", type=int, default=-1,help="few shot number,-1 means not few shot")
    parser.add_argument("--few_shot_gold_label",action="store_true",help="use gold label for few shot")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--token_entropy_size", type=int, default=1000)
    parser.add_argument("--recent_size", type=int, default=1000)
    parser.add_argument("--max_gen_len", type=int, default=20)
    parser.add_argument("--decay_ratio", type=float, default=1)
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--output_dir", type=str, default="outputs/ARC-new-eval-wo_special_token")
    parser.add_argument("--if_w_turns", action="store_true",help="whether to start a new kv cahe for a new conversation turn")
    args = parser.parse_args()

    if args.few_shot>=0:
        assert args.enable_token_entropy==False
        assert args.enable_streaming==False
        args.start_size=0
        args.recent_size=0
        args.token_entropy_size=0
        args.output_dir=os.path.join(args.output_dir,f"few_shot_{args.few_shot}_gold_{args.few_shot_gold_label}")

    if not args.enable_token_entropy:
        assert args.token_entropy_size==0
        args.token_entropy_size=0

    args.output_dir=os.path.join(args.output_dir,args.model_name_or_path.split("/")[-1])
    args.output_dir=os.path.join(args.output_dir,f"start_size_{args.start_size}_token_entropy_size_{args.token_entropy_size}_recent_size_{args.recent_size}_decay_{args.decay_ratio}_max_gen_len_{args.max_gen_len}")


    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Output path {args.output_dir} created.")
    else:
        print(f"Output path {args.output_dir} already exists.")

    # 检查文件夹下是否有文件
    if os.listdir(args.output_dir):
        print(f"Output path {args.output_dir} is not empty.")
        response = input("Do you want to rewrite the files?") 
        if response.lower() == 'yes':
            print("Rewriting files...")
        else:
            assert False

    main(args)