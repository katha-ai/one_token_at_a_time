import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_handler import BaseModelHandler
from utils.token_utils import find_subsequence

class Qwen2LLMHandler(BaseModelHandler):

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_id'],
            torch_dtype=getattr(torch, self.config['torch_dtype']),
            low_cpu_mem_usage=True,
            attn_implementation=self.config['attn_implementation'],
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_id'], use_fast=True)
        return self.model, self.tokenizer


    def get_special_token_ids(self):
        return {
            # 'image': 151646, # IMG_TOKEN_ID
            'eos': self.tokenizer.eos_token_id
            }

    def get_chunk_ranges(self, input_ids_list, task_chunk_strings, image_token_id):
                
        all_chunks_ranges = {
            'im_start_0': [], 'user': [], 'post_user_newline': [], 'sport_text': [],
            'post_sport_newline': [], 'math_text': [], 'instruction': [],
            'post_instruction_imend': [], 'post_instruction_newline': [], 'im_start_1': [],
            'assistant': [], 'post_assistant_newline': [], 'previously_generating_tokens': [], #pending
            'currently_generating_token': []
        }
        
        #the actual sequence model sees.
        tplt_im_start_str = "<|im_start|>"
        tplt_system_str = "system"
        tplt_newline_str = "\n"
        tplt_system_inst_str = "You are a helpful assistant that provides to the point answers."
        tplt_im_end_str = "<|im_end|>"
        #\n
        #im_start
        tplt_user_str = "user"
        #\n
        sport_chunk_str = task_chunk_strings['sport_text']
        math_chunk_str = task_chunk_strings['math_text']
        instruction_chunk_str = task_chunk_strings['instruction']
        #im_end
        # \n
        #im_start
        tplt_assistant_str = "assistant"
        #\n
        #previous gen tokens
        #last gen token

        im_start_ids = self.tokenizer(tplt_im_start_str, add_special_tokens=False)["input_ids"]
        system_ids = self.tokenizer(tplt_system_str, add_special_tokens=False)["input_ids"]
        newline_ids = self.tokenizer(tplt_newline_str, add_special_tokens=False)["input_ids"]
        system_instr_ids = self.tokenizer(tplt_system_inst_str, add_special_tokens=False)["input_ids"]
        user_ids = self.tokenizer(tplt_user_str, add_special_tokens=False)["input_ids"]
        sport_text_ids = self.tokenizer(sport_chunk_str, add_special_tokens=False)["input_ids"]
        math_text_ids = self.tokenizer(math_chunk_str, add_special_tokens=False)["input_ids"]
        inst_ids = self.tokenizer(instruction_chunk_str, add_special_tokens=False)["input_ids"]
        im_end_ids = self.tokenizer(tplt_im_end_str, add_special_tokens=False)["input_ids"]
        assistant_ids = self.tokenizer(tplt_assistant_str, add_special_tokens=False)["input_ids"]


        for chunk in all_chunks_ranges.keys():
            occurence = 1 #usually the first occurence, until specified
            if chunk == 'im_start_0':
                search_id = im_start_ids
                occurence = 1
            elif chunk == 'user':
                search_id = user_ids
            elif chunk == 'post_user_newline':
                search_id = newline_ids
            elif chunk == 'sport_text':
                search_id = sport_text_ids
            elif chunk == 'post_sport_newline':
                search_id = newline_ids
                occurence = 2
            elif chunk == 'math_text':
                search_id = math_text_ids
            elif chunk == 'instruction':
                search_id = inst_ids
            elif chunk == 'post_instruction_imend':
                search_id = im_end_ids
                occurence = 1
            elif chunk == 'post_instruction_newline':
                search_id = newline_ids
                occurence = 3
            elif chunk == 'im_start_1':
                search_id = im_start_ids
                occurence = 2
            elif chunk == 'assistant':
                search_id = assistant_ids
            elif chunk == 'post_assistant_newline':
                search_id = newline_ids
                occurence = 4
            else:
                continue

            start_idx = find_subsequence(input_ids_list, search_id, occurrence=occurence)
            assert start_idx is not None, f"Could not match token for {chunk}."
            end_idx = start_idx + len(search_id)
            all_chunks_ranges[chunk] = [start_idx, end_idx]

        # Find image tokens
        # img_indices = [i for i, token_id in enumerate(input_ids_list) if token_id == image_token_id]
        # if not img_indices:
        #     raise ValueError("Image token not found in input_ids")
        # all_chunks_ranges['image'] = [img_indices[0], img_indices[-1] + 1]

        return all_chunks_ranges
    
    def build_inputs(self, task_content):
        """
        Takes the standard 'task_content' dict and builds the
        LLaVA-specific conversation.
        """
        # 1. Get content from the standard dict
        full_prompt_text = task_content['full_prompt']
        # has_image = task_content['has_image']
        
        # 2. Build the LLaVA-specific 'conv' list
        conv = [{
            "role": "user",
            "content": full_prompt_text,
        }]
        
        # 3. Apply chat template and tokenize
        prompt_str = self.tokenizer.apply_chat_template(conv,
                                                        tokenize=False,
                                                        add_generation_prompt=True)

        removal_chunk = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        prompt_str = prompt_str.replace(removal_chunk, "")
        
        model_inputs = self.tokenizer(text = [prompt_str],
                                      return_tensors="pt",
                                      return_offsets_mapping=True)
        input_ids_list = model_inputs["input_ids"][0].tolist()
        return model_inputs, input_ids_list
    

    def get_tokenizer(self):
        """Returns the model's tokenizer object (for decoding, etc.)."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        return self.tokenizer

    def get_attention_layers(self):
        """Returns the list of layer modules to register hooks on."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.model.layers

    def get_layer_indices(self):
        """Returns a list of layer indices (e.g., list(range(32)))."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        return list(range(self.model.config.num_hidden_layers))
    
    def get_num_heads(self):
        """Returns the number of heads for the model."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.config.num_attention_heads