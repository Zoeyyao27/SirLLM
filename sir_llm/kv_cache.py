import torch
import numpy as np
import random


def slice2d(x, start=None, end=None,id_list=None):
    if id_list is not None:
        return x[:, :,id_list, ...]
    return x[:, :, start:end, ...]


def slice3d(x, start=None, end=None,id_list=None):
    if id_list is not None:
        return x[:, :, :, id_list, ...]
    return x[:, :, :, start:end, ...]


def slice1d(x, start=None, end=None,id_list=None):
    if id_list is not None:
        return x[:, id_list, ...]
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        token_entropy_size=1000,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}, {token_entropy_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.token_entropy_size = token_entropy_size
        self.cache_size = start_size + recent_size + token_entropy_size
        print(f"cache_size: {self.cache_size}")
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim) # past_key_values[0][0] is the first key
        if seq_len <= self.cache_size:
            return past_key_values
            
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len), 
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def _top_n_indices(self,lst, n):
        # Sort the list with indices based on values
        if n<0:
            return []
        sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
        # Return the top N indices
        return sorted_indices[:n]
    
    def evict_for_space_token_entropy(self, past_key_values,token_entropy, num_coming):

        if past_key_values is None:
            return [past_key_values,token_entropy]
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
    
        if seq_len + num_coming <= self.cache_size:
            return [past_key_values,token_entropy]

        
        if self.token_entropy_size+self.start_size+self.recent_size+num_coming>self.cache_size:
            token_entropy_size=self.token_entropy_size-num_coming
        else:
            token_entropy_size=self.token_entropy_size

         
        save_list=self._top_n_indices(token_entropy, token_entropy_size)

        save_list= sorted(save_list)

        save_list=list(set(save_list+list(range(self.start_size))))
        for i in range(seq_len-1,0,-1):
            if len(save_list)<self.cache_size - num_coming:
                if i not in save_list:
                    # make the full use of the cache
                    save_list.append(i)
            else:
                if len(save_list)>self.cache_size - num_coming: #because add first special token
                    assert False ,f"len(save_list):{len(save_list)},self.cache_size:{self.cache_size},num_coming:{num_coming},seq_len:{seq_len},token_entropy_size:{token_entropy_size},token_entropy:{token_entropy}"
                    save_list=save_list[:self.cache_size - num_coming]
                break

        save_list= sorted(save_list)
        #
        if token_entropy:
            token_entropy_saved = [token_entropy[i] for i in save_list]
        else:
            token_entropy_saved = None
        #assert False
        
        assert len(save_list)+num_coming<=self.cache_size, f"len(save_list):{len(save_list)},num_coming:{num_coming},self.cache_size:{self.cache_size},recent_size:{self.recent_size},token_entropy_size:{token_entropy_size},start_size:{self.start_size},seq_len:{seq_len},token_entropy_size:{token_entropy_size},token_entropy:{token_entropy}"

        
        return [[
            [
                self.k_slice(k, id_list=save_list),
                self.v_slice(v, id_list=save_list),
            ]
            for k, v in past_key_values
        ],token_entropy_saved]

 

    def evict_for_space(self, past_key_values, num_coming):
        #assert False, "not implemented"
        #num_coming is the number of tokens in the current batch + max_gen_len
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return [past_key_values]
        else:
            return [[
                [
                    torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            self.k_slice(
                                k, seq_len - self.recent_size + num_coming, seq_len
                            ),
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            self.v_slice(v, 0, self.start_size),
                            self.v_slice(
                                v, seq_len - self.recent_size + num_coming, seq_len
                            ),
                        ],
                        dim=self.v_seq_dim,
                    ),
                ]
                for k, v in past_key_values
            ]]
    