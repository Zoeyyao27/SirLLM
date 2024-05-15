# 🎩SirLLM: Streaming Infinite Retentive LLM

We introduce Streaming Infinite Retentive LLM (SirLLM), which utilizes the Token Entropy metric and a memory decay mechanism to filter key phrases, endowing LLMs with both long-lasting and flexible memory.

![Image text](https://github.com/Zoeyyao27/SirLLM/blob/main/image/sirllm.png)


# Get Started
## 🛠️ Preparation
```
pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
```
## 🎩 Run SirLLM

### 👉🏻 Grocery Shopping Dataset
```
CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama_concate_question_new_eval.py  \ 
    --start_size 4 --data_root "data/grocery_keys" \
    --model_name_or_path "01-ai/Yi-6B-Chat"   \
    --enable_streaming --token_entropy_size 1020 \
    --recent_size 0 --enable_token_entropy \
    --output_dir "outputs/keys" --decay_ratio 1
```
### 👉🏻 DailyDialog Dataset
```
CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama_concate_question_new_eval.py \
    --if_w_turns  --start_size 4 \
    --data_root "data/dailydialog" \
    --model_name_or_path "01-ai/Yi-6B-Chat" \
    --enable_streaming --token_entropy_size 508 \
    --recent_size 0 --enable_token_entropy \
    --output_dir "outputs/dailydialog" \
    --decay_ratio 0.7 
```
### 👉🏻 Rock-paper-scissors
```
CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama_concate_question_new_eval.py \
    --start_size 4 --data_root "data/rock_paper_scissors" \
    --model_name_or_path "01-ai/Yi-6B-Chat" \
    --enable_streaming --token_entropy_size 1020 \
    --recent_size 0 --enable_token_entropy \
    --output_dir "outputs/rock_paper_scissors" \
    --decay_ratio 0.9 

```

# Acknowledgement
💐Many thanks to [Streamllm](https://github.com/mit-han-lab/streaming-llm). Some portions of this codebase were inspired by or directly borrowed from the  [Streamllm](https://github.com/mit-han-lab/streaming-llm). Their contributions have been invaluable in the development of this project.

# Citing 🎩SirLLM


