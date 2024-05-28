from tqdm import tqdm
from typing import List, Dict, Union, Tuple, Literal
import torch.nn.functional as F
import torch
import random
import numpy as np
from rouge import Rouge



def slice2d(x, start=None, end=None,id_list=None):
    if id_list is not None:
        return x[:, :,id_list, ...]
    return x[:, :, start:end, ...]



class Evaluator:
    def __init__(self, model, tokenizer,args ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.decay_ratio=args.decay_ratio
        if "Yi" in self.args.model_name_or_path and "Chat" in self.args.model_name_or_path:
            messages = [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""},
            ]
            template_text=self._construct_chat_template(messages,add_generation_prompt=True)
            self.template_ids=self.tokenizer(template_text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze().tolist()
        else:
            self.template_ids=None


    @torch.no_grad()
    def _cal_selfinfo(self, total_ids,logits):
        if logits == None:
            return None
        probs = torch.softmax(logits, dim=-1)
        token_entropy = -torch.log(probs)#[bz,l+1,vocab_size]
        total_ids_expaned = torch.tensor(total_ids).unsqueeze(-1).unsqueeze(0).to(token_entropy.device)
        token_entropy_list=token_entropy.gather(-1, total_ids_expaned).squeeze(-1).squeeze(0).tolist() #get the token_entropy of the next token
        
        #the logits of the current token is the token_entropy of the next token
        max_token_entropy=max(token_entropy_list)
        token_entropy_list=[max_token_entropy]+token_entropy_list #add the token_entropy of the first token
        token_entropy_list= token_entropy_list[:-1] 
        return token_entropy_list

    @torch.no_grad()
    def _greedy_generate_token_entropy(self, input_ids,continue_len,past_key_values,token_entropy):
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        past_logits = outputs.logits
        logits = F.log_softmax(
            outputs.logits, dim=-1
        )

        continue_logits = logits[:, -continue_len-1:-1, :]
        greedy_tokens_ids = continue_logits.argmax(dim=-1)
        cont_toks_ids=input_ids[:,-continue_len:]

        greedy_tokens=self.tokenizer.decode(greedy_tokens_ids.squeeze().tolist())
        greedy_tokens=greedy_tokens.replace(" ", "")
        cont_toks=self.tokenizer.decode(cont_toks_ids.squeeze().tolist())
        cont_toks=cont_toks.replace(" ", "")
        max_equal = (greedy_tokens == cont_toks)

        logits = torch.gather(continue_logits, 2, cont_toks_ids.unsqueeze(-1)).squeeze(
            -1
        )  # [1, seq]


        answer = (float(logits.squeeze(0)[0]), bool(max_equal))

        total_ids=input_ids.squeeze().tolist()

        token_entropy = self._cal_selfinfo(total_ids,past_logits)

        return  past_key_values,token_entropy, answer

    @torch.no_grad()
    def _greedy_generate(self, input_ids,continue_len,past_key_values,use_cache=True):
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )


        past_key_values = outputs.past_key_values

        logits = F.log_softmax(
            outputs.logits, dim=-1
        )

        continue_logits = logits[:, -continue_len-1:-1, :]
        greedy_tokens_ids = continue_logits.argmax(dim=-1)
        cont_toks_ids=input_ids[:,-continue_len:]

        greedy_tokens=self.tokenizer.decode(greedy_tokens_ids.squeeze().tolist())
        greedy_tokens=greedy_tokens.replace(" ", "")
        cont_toks=self.tokenizer.decode(cont_toks_ids.squeeze().tolist())
        cont_toks=cont_toks.replace(" ", "")
        max_equal = (greedy_tokens == cont_toks)

        logits = torch.gather(continue_logits, 2, cont_toks_ids.unsqueeze(-1)).squeeze(
            -1
        )  # [1, seq]


        ###only choose the first logits as answer
        answer = (float(logits.squeeze(0)[0]), bool(max_equal))

        #print("!!!hhhanswer",answer) 

        return past_key_values, answer
    
    def _construct_chat_template(self,messages,add_generation_prompt=True):
        if "Yi" in self.args.model_name_or_path and "Chat" in self.args.model_name_or_path:
            text=""
            for message in messages:
                text=text+"<|im_start|>"+message["role"]+"\n"+message["content"]+"<|im_end|>"+"\n"
            if add_generation_prompt:
                text=text+"<|im_start|>assistant\n"
            return text

        
    @torch.no_grad()
    def _encode_pair(
        self,context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        # whole_enc = self.tokenizer(context + continuation, return_tensors="pt").input_ids
        
        if "Yi" in self.args.model_name_or_path and "Chat" in self.args.model_name_or_path:
            # Prompt content: "hi"
            messages = [
                {"role": "user", "content": context},
                {"role": "assistant", "content": continuation}
            ]

            text=self._construct_chat_template(messages,add_generation_prompt=False)
            whole_enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
            #assert False
        
        else:
            whole_enc = self.tokenizer(context + continuation, add_special_tokens=False, return_tensors="pt").input_ids

        if "Yi" in self.args.model_name_or_path and "Chat" in self.args.model_name_or_path:
            # Prompt content: "hi"
            messages = [
                {"role": "user", "content": context}
            ]
            text=self._construct_chat_template(messages,add_generation_prompt=True)
            context_enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        else:            
            context_enc = self.tokenizer(context, add_special_tokens=False, return_tensors="pt").input_ids
            

        context_enc = context_enc.to(self.model.device)
        whole_enc = whole_enc.to(self.model.device)
        
        context_enc_len = context_enc.shape[1]
        continuation_enc=whole_enc[:,context_enc_len:]

        return context_enc, continuation_enc

    @torch.no_grad()
    def _encode_single(
        self,context: str
    ) -> Tuple[List[int], List[int]]:

        if "Yi" in self.args.model_name_or_path and "Chat" in self.args.model_name_or_path:
            # Prompt content: "hi"
            messages = [
                {"role": "user", "content": context}
            ]
            text=self._construct_chat_template(messages,add_generation_prompt=True)
            context_enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        else:            
            context_enc = self.tokenizer(context, add_special_tokens=False, return_tensors="pt").input_ids
            

        context_enc = context_enc.to(self.model.device)

        return context_enc



    @torch.no_grad()
    def streaming_inference(self,prompts, kv_cache=None):
        past_key_values = None
        correct=0
        total_genearted_text=[]

        pbar = tqdm(total=len(prompts))
        for req in prompts:
            new_reqs = []
            gold_ans_id = req["answer_id"]
            if isinstance(req, dict):

                if "Yi" in self.args.model_name_or_path and "Chat" in self.args.model_name_or_path:
                    context = req["question"] 
                    #assert False
                else:
                    context ="USER: " + req["question"] + "\n\nASSISTANT: "    

            choices = req["choices"]
            for continuation in choices:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
                new_reqs.append(((context, continuation), context_enc, continuation_enc))


            choice_past_key_values = []
            seq_len_list=[]
            answers = []
            answer_logits=[]

            if kv_cache is not None and past_key_values is not None:
                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                temp= kv_cache.evict_for_space(past_key_values, seq_len)
                past_key_values=temp[0]

            for _, context_enc, continuation_enc in new_reqs:

                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                continue_len=continuation_enc.shape[1]
                choice_past_key_value, answer=self._greedy_generate(
                    input_ids,continue_len=continue_len,past_key_values=past_key_values
                    )
                choice_past_key_values.append(choice_past_key_value)
                seq_len_list.append(seq_len)
                answers.append(answer[1])
                answer_logits.append(answer[0])

            if True in answers:
                choose_idx=answers.index(True)
            else:
                choose_idx=answer_logits.index(max(answer_logits))

            if gold_ans_id == choose_idx:
                #print("!!correct")
                correct+=1          

            past_key_values=choice_past_key_values[choose_idx]
            total_genearted_text.append({"pred_ans":choose_idx,"gold_ans":gold_ans_id,"prompt":context,"choices":choices})

            pbar.update(1)
        pbar.close()
        print("!!total",len(prompts))
        print("!!accuracy",correct/len(prompts))
        return correct/len(prompts),total_genearted_text 


    @torch.no_grad()
    def streaming_inference_w_turns(self,prompts, kv_cache=None):
        past_key_values = None
        correct=0
        total_genearted_text=[]

        pbar = tqdm(total=len(prompts))

        previous_turn=0
        for req in prompts:
            q_id=req["id"]
            current_turn=int(q_id.split("_")[0])

            if current_turn!=previous_turn:
                past_key_values = None

            previous_turn=current_turn


            new_reqs = []
            gold_ans_id = req["answer_id"]
            if isinstance(req, dict):

                if "Yi" in self.args.model_name_or_path and "Chat" in self.args.model_name_or_path:
                    context = req["question"] 
                    #assert False
                else:
                    context ="USER: " + req["question"] + "\n\nASSISTANT: "     
            choices = req["choices"]
            for continuation in choices:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
                new_reqs.append(((context, continuation), context_enc, continuation_enc))

            choice_past_key_values = []
            seq_len_list=[]
            answers = []
            answer_logits=[]

            if kv_cache is not None and past_key_values is not None:
                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                temp= kv_cache.evict_for_space(past_key_values, seq_len)
                past_key_values=temp[0]

            for _, context_enc, continuation_enc in new_reqs:

                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                continue_len=continuation_enc.shape[1]
                choice_past_key_value, answer=self._greedy_generate(
                    input_ids,continue_len=continue_len,past_key_values=past_key_values
                    )
                choice_past_key_values.append(choice_past_key_value)
                seq_len_list.append(seq_len)
                answers.append(answer[1])
                answer_logits.append(answer[0])

            if True in answers:
                choose_idx=answers.index(True)

            else:
                choose_idx=answer_logits.index(max(answer_logits))
            if gold_ans_id == choose_idx:
                #print("!!correct")
                correct+=1

            past_key_values=choice_past_key_values[choose_idx]

            total_genearted_text.append({"pred_ans":choose_idx,"gold_ans":gold_ans_id,"prompt":context,"choices":choices})

            pbar.update(1)
        pbar.close()
        print("!!total",len(prompts))
        print("!!accuracy",correct/len(prompts))
        return correct/len(prompts),total_genearted_text 

    @torch.no_grad()
    def streaming_inference_for_rps(self,prompts, kv_cache=None):
        past_key_values = None
        win=0
        tie=0
        lose=0
        exact_match=0
        total_genearted_text=[]

        last_turn_user_choice=None
        # last_turn_result=None

        pbar = tqdm(total=len(prompts))
        for req in prompts:
            new_reqs = []
            gold_ans_id = req["answer_id"]
            # gold_ans=req["answer"]
            user_choice_id=req["user_choice_id"]
            user_choice=req["user_choice"]
            if isinstance(req, dict):
                #prompt
                if last_turn_user_choice is not None:
                    question =f"I chose [{last_turn_user_choice}]. Analyze my preferences to maximize your winning rate and proceed to the next round. Choices: (A) rock (B) paper (C) scissors. You choose:"
                else: 
                    question="Let's play a game of Rock-Paper-Scissors. Choose one option from: (A) rock (B) paper (C) scissors. " 


                if "Yi" in self.args.model_name_or_path and "Chat" in self.args.model_name_or_path:
                    context = question 
                    #assert False
                else:
                    context ="USER: " + question + "\n\nASSISTANT: "   

            choices_letter=req["choices_letter"]
            for continuation in choices_letter: ###choose letter
                context_enc, continuation_enc = self._encode_pair(context, continuation)
                new_reqs.append(((context, continuation), context_enc, continuation_enc))


            choice_past_key_values = []
            seq_len_list=[]
            answers = []
            answer_logits=[]

            if kv_cache is not None and past_key_values is not None:
                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                temp= kv_cache.evict_for_space(past_key_values, seq_len)
                past_key_values=temp[0]

            for _, context_enc, continuation_enc in new_reqs:

                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                continue_len=continuation_enc.shape[1]
                choice_past_key_value, answer=self._greedy_generate(
                    input_ids,continue_len=continue_len,past_key_values=past_key_values
                    )
                choice_past_key_values.append(choice_past_key_value)
                seq_len_list.append(seq_len)
                answers.append(answer[1])
                answer_logits.append(answer[0])


            choose_idx=answer_logits.index(max(answer_logits))
            if gold_ans_id == choose_idx:
                #print("!!win")
                win+=1
                last_turn_result="win"
            if choose_idx==user_choice_id:
                tie+=1
                last_turn_result="tie"
            if gold_ans_id != choose_idx and choose_idx!=user_choice_id:
                lose+=1
                last_turn_result="lose"

            
            past_key_values=choice_past_key_values[choose_idx]


            total_genearted_text.append({"pred_ans":choose_idx,"gold_ans":gold_ans_id,"prompt":context,"choices":choices_letter})
            last_turn_user_choice=user_choice

            pbar.update(1)
        pbar.close()
        print("!!total",len(prompts))
        print("!!win rate",win/len(prompts))
        print("!!tie rate",tie/len(prompts))
        print("!!lose rate",lose/len(prompts))
        print("!!exact_match rate",exact_match/len(prompts))
        res={}
        res["win_rate"]=win/len(prompts)
        res["tie_rate"]=tie/len(prompts)
        res["lose_rate"]=lose/len(prompts)
        res["exact_match_rate"]=exact_match/len(prompts)
        return res,total_genearted_text
     
    @torch.no_grad()
    def streaming_inference_token_entropy_for_rps(self,prompts, kv_cache=None):
        past_key_values = None
        token_entropy = None
        win=0
        tie=0
        lose=0
        exact_match=0
        total_genearted_text=[]
        #total_special_token_mask=None
        pbar = tqdm(total=len(prompts))

        last_turn_user_choice=None
        last_turn_result=None

        for id,req in enumerate(prompts):
            new_reqs = []
            gold_ans_id = req["answer_id"]
            gold_ans=req["answer"]
            user_choice_id=req["user_choice_id"]
            user_choice=req["user_choice"]
            if isinstance(req, dict):

                #prompt 
                if last_turn_user_choice is not None:
                    question =f"I chose [{last_turn_user_choice}]. Analyze my preferences to maximize your winning rate and proceed to the next round. Choices: (A) rock (B) paper (C) scissors. You choose:"
                else: 
                    question="Let's play a game of Rock-Paper-Scissors. Choose one option from: (A) rock (B) paper (C) scissors. " 


                if "Yi" in self.args.model_name_or_path and "Chat" in self.args.model_name_or_path:
                    context = question 
                    #assert False
                else:
                    context ="USER: " + question + "\n\nASSISTANT: "   
            
            #print("!!!context",context)
            choices = req["choices"] ###choose text
            choices_letter=req["choices_letter"]
            for continuation in choices_letter: ###choose letter
                context_enc, continuation_enc = self._encode_pair(context, continuation)

                new_reqs.append(((context, continuation), context_enc, continuation_enc))

            choice_past_key_values = []
            choice_token_entropys = []
            seq_len_list=[]
            answers = []
            answer_logits=[]


            if kv_cache is not None and token_entropy is not None:

                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                temp= kv_cache.evict_for_space_token_entropy(past_key_values,token_entropy, seq_len)
                past_key_values=temp[0]
                token_entropy=temp[1]

                token_entropy = np.array(token_entropy) 
                scaled_token_entropy = token_entropy * self.decay_ratio   
                token_entropy = scaled_token_entropy.tolist() 


            for _, context_enc, continuation_enc in new_reqs:

                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                continue_len=continuation_enc.shape[1]
                choice_past_key_value,choice_token_entropy, answer=self._greedy_generate_token_entropy(
                    input_ids,continue_len=continue_len,past_key_values=past_key_values,token_entropy=token_entropy
                    )
                choice_past_key_values.append(choice_past_key_value)
                choice_token_entropys.append(choice_token_entropy)

                seq_len_list.append(seq_len)
                answers.append(answer[1])
                answer_logits.append(answer[0])

            choose_idx=answer_logits.index(max(answer_logits))
            if gold_ans_id == choose_idx:
                print("!!win")
                win+=1
                last_turn_result="win"
            if choose_idx==user_choice_id:
                tie+=1
                last_turn_result="tie"
            if gold_ans_id != choose_idx and choose_idx!=user_choice_id:
                lose+=1
                last_turn_result="lose"


            past_key_values=choice_past_key_values[choose_idx]

            if token_entropy:
                token_entropy +=choice_token_entropys[choose_idx]
            else:
                token_entropy = choice_token_entropys[choose_idx]
        
            total_genearted_text.append({"pred_ans":choose_idx,"gold_ans":gold_ans_id,"prompt":context,"choices":choices_letter})
            last_turn_user_choice=user_choice

            pbar.update(1)
        pbar.close()
        print("!!total",len(prompts))
        print("!!win rate",win/len(prompts))
        print("!!tie rate",tie/len(prompts))
        print("!!lose rate",lose/len(prompts))
        print("!!exact_match rate",exact_match/len(prompts))
        res={}
        res["win_rate"]=win/len(prompts)
        res["tie_rate"]=tie/len(prompts)
        res["lose_rate"]=lose/len(prompts)
        res["exact_match_rate"]=exact_match/len(prompts)
        return res,total_genearted_text
    
    @torch.no_grad()
    def streaming_inference_token_entropy(self,prompts, kv_cache=None):
        past_key_values = None
        token_entropy = None
        correct=0
        total_genearted_text=[]
        pbar = tqdm(total=len(prompts))

        for req in prompts:
            new_reqs = []
            gold_ans_id = req["answer_id"]
            if isinstance(req, dict):

                if "Yi" in self.args.model_name_or_path and "Chat" in self.args.model_name_or_path:
                    context = req["question"] 
                    #assert False
                else:
                    context ="USER: " + req["question"] + "\n\nASSISTANT: "    
            choices = req["choices"]
            for continuation in choices:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
                new_reqs.append(((context, continuation), context_enc, continuation_enc))

            choice_past_key_values = []
            choice_token_entropys = []
            seq_len_list=[]
            answers = []
            answer_logits=[]

            if kv_cache is not None and token_entropy is not None:
                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                temp= kv_cache.evict_for_space_token_entropy(past_key_values,token_entropy, seq_len)
                past_key_values=temp[0]
                token_entropy=temp[1]

                token_entropy = np.array(token_entropy) 
                scaled_token_entropy = token_entropy * self.decay_ratio   
                token_entropy = scaled_token_entropy.tolist() 

            for _, context_enc, continuation_enc in new_reqs:

                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                continue_len=continuation_enc.shape[1]
                choice_past_key_value,choice_token_entropy, answer=self._greedy_generate_token_entropy(
                    input_ids,continue_len=continue_len,past_key_values=past_key_values,token_entropy=token_entropy
                    )
                choice_past_key_values.append(choice_past_key_value)
                choice_token_entropys.append(choice_token_entropy)
                seq_len_list.append(seq_len)
                answers.append(answer[1])
                answer_logits.append(answer[0])

            if True in answers:
                choose_idx=answers.index(True)
            else:
                choose_idx=answer_logits.index(max(answer_logits))
                #random.randint(0,3)
            if gold_ans_id == choose_idx:
                print("!!correct")
                correct+=1

            past_key_values=choice_past_key_values[choose_idx]

            if token_entropy:
                token_entropy +=choice_token_entropys[choose_idx]
            else:
                token_entropy = choice_token_entropys[choose_idx]
        

            total_genearted_text.append({"pred_ans":choose_idx,"gold_ans":gold_ans_id,"prompt":context,"choices":choices})

            pbar.update(1)
        pbar.close()
        print("!!total",len(prompts))
        print("!!accuracy",correct/len(prompts))
        return correct/len(prompts),total_genearted_text       

    @torch.no_grad()
    def streaming_inference_token_entropy_w_turns(self,prompts, kv_cache=None):
        past_key_values = None
        token_entropy = None
        # total_special_token_mask=None
        correct=0
        total_genearted_text=[]

        pbar = tqdm(total=len(prompts))

        previous_turn=0
        for req in prompts:
            q_id=req["id"]
            current_turn=int(q_id.split("_")[0])

            if current_turn!=previous_turn:

                past_key_values = None
                token_entropy = None
                # total_special_token_mask=None
            previous_turn=current_turn
            new_reqs = []
            gold_ans_id = req["answer_id"]
            context ="USER: " + req["question"] + "\n\nASSISTANT: "       
            choices = req["choices"]
            for continuation in choices:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
                #new_reqs.append(((context, continuation), torch.tensor(context_enc), torch.tensor(continuation_enc)))
                new_reqs.append(((context, continuation), context_enc, continuation_enc))

            choice_past_key_values = []
            choice_token_entropys = []
            seq_len_list=[]
            answers = []
            answer_logits=[]

            if kv_cache is not None and token_entropy is not None:
                #space_needed = seq_len_list[choose_idx] #+ self.args.max_gen_len
                # if clean_flag:
                #     space_needed = space_needed - sum(special_token_mask)
                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                temp= kv_cache.evict_for_space_token_entropy(past_key_values,token_entropy, seq_len)
                past_key_values=temp[0]
                token_entropy=temp[1]
                #total_special_token_mask=temp[2]
                token_entropy = np.array(token_entropy) 
                scaled_token_entropy = token_entropy * self.decay_ratio   
                token_entropy = scaled_token_entropy.tolist() 

            for _, context_enc, continuation_enc in new_reqs:
                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                continue_len=continuation_enc.shape[1]
                choice_past_key_value,choice_token_entropy, answer=self._greedy_generate_token_entropy(
                    input_ids,continue_len=continue_len,past_key_values=past_key_values,token_entropy=token_entropy
                    )
                choice_past_key_values.append(choice_past_key_value)
                choice_token_entropys.append(choice_token_entropy)

                seq_len_list.append(seq_len)
                answers.append(answer[1])
                answer_logits.append(answer[0])

            if True in answers:
                choose_idx=answers.index(True)
                # if gold_ans_id == choose_idx:
                #     print("!!correct")
                #     correct+=1
            else:
                choose_idx=answer_logits.index(max(answer_logits))
                #random.randint(0,3)
            if gold_ans_id == choose_idx:
                print("!!correct")
                correct+=1

            past_key_values=choice_past_key_values[choose_idx]

            if token_entropy:
                token_entropy +=choice_token_entropys[choose_idx]
            else:
                token_entropy = choice_token_entropys[choose_idx]

            # total_special_token_mask=[0 for i in range(len(token_entropy))]
            total_genearted_text.append({"pred_ans":choose_idx,"gold_ans":gold_ans_id,"prompt":context,"choices":choices})

            pbar.update(1)
        pbar.close()
        print("!!total",len(prompts))
        print("!!accuracy",correct/len(prompts))
        return correct/len(prompts),total_genearted_text       
    

    @torch.no_grad()
    def inference_few_shot(self,prompts,few_shot,gold_label):
        correct=0
        total_genearted_text=[]
        pred_ans=[]
        pbar = tqdm(total=len(prompts))
        for id,req in enumerate(prompts):
            new_reqs = []
            gold_ans_id = req["answer_id"]
            if "vicuna" in self.args.model_name_or_path:
                context=""
                if id>=few_shot: 
                    for i in range(id-few_shot,id):
                        if gold_label:
                            context+="USER: "+ prompts[i]["question"] + "\n\nASSISTANT: " + prompts[i]["choices"][prompts[i]["answer_id"]]+ "\n"
                        else:
                            context += "USER: "+ prompts[i]["question"] + "\n\nASSISTANT: " + prompts[i]["choices"][pred_ans[i]]+ "\n"
                context +="USER: "+ req["question"]+  "\n\nASSISTANT: "
            else:
                raise ValueError(f"{self.args.model_name_or_path} is not supported!")
            #print("!!!context",context)

            choices = req["choices"]
            for continuation in choices:
                context_enc, continuation_enc,_ = self._encode_pair(context, continuation)
                #new_reqs.append(((context, continuation), torch.tensor(context_enc), torch.tensor(continuation_enc)))
                new_reqs.append(((context, continuation), context_enc, continuation_enc))
                # print("!!!context",context)
                # print("!!!continuation",continuation)
                # print("!!!continuation_enc",continuation_enc)

            answers = []
            answer_logits=[]
            for _, context_enc, continuation_enc in new_reqs:

                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)

                continue_len=continuation_enc.shape[1]
                _, answer=self._greedy_generate(
                    input_ids,continue_len=continue_len,past_key_values=None,use_cache=False
                    )

                answers.append(answer[1])
                answer_logits.append(answer[0])

            if True in answers:
                choose_idx=answers.index(True)
                # assert False
                # if gold_ans_id == choose_idx:
                #     print("!!correct")#,gold_ans_id)
                #     correct+=1
                # else:
                #     print("!!wrong",gold_ans_id,choose_idx)
            else:
                choose_idx=answer_logits.index(max(answer_logits))

            if gold_ans_id == choose_idx:
                print("!!correct")
                correct+=1

            pred_ans.append(choose_idx)


            total_genearted_text.append({"pred_ans":choose_idx,"gold_ans":gold_ans_id,"prompt":context,"choices":choices})

            pbar.update(1)
        pbar.close()
        print("!!total",len(prompts))
        print("!!accuracy",correct/len(prompts))
        return correct/len(prompts),total_genearted_text 
    


    @torch.no_grad()
    def streaming_inference_token_entropy_for_keys(self,prompts, kv_cache=None):
        past_key_values = None
        token_entropy = None
        # total_special_token_mask=None
        correct=0
        correct_key=0
        total_arc_num=0
        total_key_num=0
        total_genearted_text=[]

        pbar = tqdm(total=len(prompts))

        previous_turn=0
        for req in prompts:
            q_id=req["id"]
            current_turn=int(q_id.split("_")[-1])

            #print("!!!current_turn",current_turn)

            if current_turn!=previous_turn:

                past_key_values = None
                token_entropy = None
                # total_special_token_mask=None
            previous_turn=current_turn
            new_reqs = []
            gold_ans_id = req["answer_id"]
            context ="USER: " + req["question"] + "\n\nASSISTANT: "       
            choices = req["choices"]
            for continuation in choices:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
                #new_reqs.append(((context, continuation), torch.tensor(context_enc), torch.tensor(continuation_enc)))
                new_reqs.append(((context, continuation), context_enc, continuation_enc))

            choice_past_key_values = []
            choice_token_entropys = []
            seq_len_list=[]
            answers = []
            answer_logits=[]
    
            if kv_cache is not None and token_entropy is not None:
                #space_needed = seq_len_list[choose_idx] #+ self.args.max_gen_len
                # if clean_flag:
                #     space_needed = space_needed - sum(special_token_mask)
                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                temp= kv_cache.evict_for_space_token_entropy(past_key_values,token_entropy, seq_len)
                past_key_values=temp[0]
                token_entropy=temp[1]
                #total_special_token_mask=temp[2]
                token_entropy = np.array(token_entropy) 
                scaled_token_entropy = token_entropy * self.decay_ratio   
                token_entropy = scaled_token_entropy.tolist() 

            for _, context_enc, continuation_enc in new_reqs:
                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                continue_len=continuation_enc.shape[1]
                choice_past_key_value,choice_token_entropy, answer=self._greedy_generate_token_entropy(
                    input_ids,continue_len=continue_len,past_key_values=past_key_values,token_entropy=token_entropy
                    )
                choice_past_key_values.append(choice_past_key_value)
                choice_token_entropys.append(choice_token_entropy)

                seq_len_list.append(seq_len)
                answers.append(answer[1])
                answer_logits.append(answer[0])

            if True in answers:
                choose_idx=answers.index(True)
                # if gold_ans_id == choose_idx:
                #     print("!!correct")
                #     correct+=1
            else:
                choose_idx=answer_logits.index(max(answer_logits))
                #random.randint(0,3)
            if "key" not in q_id and "answer" not in q_id:
                total_arc_num+=1
                if gold_ans_id == choose_idx:
                    print("!!correct")
                    correct+=1
            elif "answer" in q_id:
                total_key_num+=1
                if gold_ans_id == choose_idx:
                    print("!!key_correct")
                    correct_key+=1

            past_key_values=choice_past_key_values[choose_idx]

            if token_entropy:
                token_entropy +=choice_token_entropys[choose_idx]
            else:
                token_entropy = choice_token_entropys[choose_idx]

            # total_special_token_mask=[0 for i in range(len(token_entropy))]
            total_genearted_text.append({"pred_ans":choose_idx,"gold_ans":gold_ans_id,"prompt":context,"choices":choices})

            pbar.update(1)
        pbar.close()
        print("!!total",len(prompts))
        print("!!accuracy",correct/total_arc_num)
        print("!!accuracy_key",correct_key/total_key_num)
        return correct/total_arc_num,correct_key/total_key_num,total_genearted_text,total_arc_num,total_key_num       

    @torch.no_grad()
    def streaming_inference_for_keys(self,prompts, kv_cache=None):
        past_key_values = None
        correct=0
        correct_key=0
        total_arc_num=0
        total_key_num=0
        total_genearted_text=[]

        pbar = tqdm(total=len(prompts))

        previous_turn=0
        for req in prompts:
            q_id=req["id"]
            current_turn=int(q_id.split("_")[-1])

            if current_turn!=previous_turn:
                past_key_values = None

            previous_turn=current_turn


            new_reqs = []
            gold_ans_id = req["answer_id"]
            if isinstance(req, dict):
                if "Yi" in self.args.model_name_or_path and "Chat" in self.args.model_name_or_path:
                    context = req["question"] 
                    #assert False
                else:
                    context ="USER: " + req["question"] + "\n\nASSISTANT: "    
                #context ="USER: " + req["question"] + "\n\nASSISTANT: "    
            choices = req["choices"]
            for continuation in choices:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
                #new_reqs.append(((context, continuation), torch.tensor(context_enc), torch.tensor(continuation_enc)))
                new_reqs.append(((context, continuation), context_enc, continuation_enc))
                # print("!!!context",context)
                # print("!!!continuation",continuation)
                # print("!!!continuation_enc",continuation_enc)

            choice_past_key_values = []
            seq_len_list=[]
            answers = []
            answer_logits=[]

            if kv_cache is not None and past_key_values is not None:
                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]

                temp= kv_cache.evict_for_space(past_key_values, seq_len)
                past_key_values=temp[0]
            #print("!!!past_key_values_after",past_key_values[0][0].shape)
                
            for _, context_enc, continuation_enc in new_reqs:

                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)
                seq_len = input_ids.shape[1]
                continue_len=continuation_enc.shape[1]
                choice_past_key_value, answer=self._greedy_generate(
                    input_ids,continue_len=continue_len,past_key_values=past_key_values
                    )
                choice_past_key_values.append(choice_past_key_value)
                seq_len_list.append(seq_len)
                answers.append(answer[1])
                answer_logits.append(answer[0])

            if True in answers:
                choose_idx=answers.index(True)
                # assert False
                # if gold_ans_id == choose_idx:
                #     print("!!correct")
                #     correct+=1
            else:
                choose_idx=answer_logits.index(max(answer_logits))
            if "key" not in q_id and "answer" not in q_id:
                total_arc_num+=1
                if gold_ans_id == choose_idx:
                    print("!!correct")
                    correct+=1
            elif "answer" in q_id:
                total_key_num+=1
                if gold_ans_id == choose_idx:
                    print("!!key_correct")
                    correct_key+=1

            # choose_idx=answer_logits.index(max(answer_logits))
                #random.randint(0,3)

            past_key_values=choice_past_key_values[choose_idx]
            # print("!!!past_key_values",past_key_values[0][0].shape)





            total_genearted_text.append({"pred_ans":choose_idx,"gold_ans":gold_ans_id,"prompt":context,"choices":choices})

            pbar.update(1)
        pbar.close()
        print("!!total",len(prompts))
        print("!!accuracy",correct/total_arc_num)
        print("!!accuracy_key",correct_key/total_key_num)
        return correct/total_arc_num,correct_key/total_key_num,total_genearted_text,total_arc_num,total_key_num       


    
    def rouge_score(self,prediction, ground_truth, **kwargs):
        rouge = Rouge()
        try:
            scores = rouge.get_scores(prediction, ground_truth, avg=True)
        except:
            return 0.0
        return scores["rouge-l"]["f"]



    
    @torch.no_grad()
    def inference_few_shot_for_keys(self,prompts,few_shot,gold_label):
        correct=0
        correct_key=0
        total_arc_num=0
        total_key_num=0
        total_genearted_text=[]
        pred_ans=[]
        pbar = tqdm(total=len(prompts))

        previous_turn=0
        for id,req in enumerate(prompts):
            q_id=req["id"]
            new_reqs = []
            gold_ans_id = req["answer_id"]
            if isinstance(req, dict):
                context=""
                if id >=few_shot:
                    for i in range(id-few_shot,id):
                        if gold_label:
                            context += "USER: "+ prompts[i]["question"] + "\n\nASSISTANT: " + prompts[i]["choices"][prompts[i]["answer_id"]]+ "\n" 
                        else:
                            context += "USER: "+ prompts[i]["question"] + "\n\nASSISTANT: " + prompts[i]["choices"][pred_ans[i]]+ "\n"
                            
                if "Yi" in self.args.model_name_or_path and "Chat" in self.args.model_name_or_path:
                    context += req["question"] 
                    #assert False
                else:
                    context +="USER: " + req["question"] + "\n\nASSISTANT: "    
                #context ="USER: " + req["question"] + "\n\nASSISTANT: "    
            choices = req["choices"]
            for continuation in choices:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
                #new_reqs.append(((context, continuation), torch.tensor(context_enc), torch.tensor(continuation_enc)))
                new_reqs.append(((context, continuation), context_enc, continuation_enc))


            answers = []
            answer_logits=[]
            for _, context_enc, continuation_enc in new_reqs:

                input_ids= torch.cat((context_enc, continuation_enc), dim=-1)
                input_ids = input_ids.to(self.model.device)

                continue_len=continuation_enc.shape[1]
                _, answer=self._greedy_generate(
                    input_ids,continue_len=continue_len,past_key_values=None,use_cache=False
                    )


                answers.append(answer[1])
                answer_logits.append(answer[0])

            if True in answers:
                choose_idx=answers.index(True)
            else:
                choose_idx=answer_logits.index(max(answer_logits))
            if "key" not in q_id and "answer" not in q_id:
                total_arc_num+=1
                if gold_ans_id == choose_idx:
                    print("!!correct")
                    correct+=1
            elif "answer" in q_id:
                total_key_num+=1
                if gold_ans_id == choose_idx:
                    print("!!key_correct")
                    correct_key+=1

            pred_ans.append(choose_idx)
            total_genearted_text.append({"pred_ans":choose_idx,"gold_ans":gold_ans_id,"prompt":context,"choices":choices})

            pbar.update(1)
        pbar.close()
        print("!!total",len(prompts))
        print("!!accuracy",correct/total_arc_num)
        print("!!accuracy_key",correct_key/total_key_num)
        return correct/total_arc_num,correct_key/total_key_num,total_genearted_text,total_arc_num,total_key_num       







