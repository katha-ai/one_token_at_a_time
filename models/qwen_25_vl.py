import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from .base_handler import BaseModelHandler
from utils.token_utils import find_subsequence

class QwenVLHandler(BaseModelHandler):

    def build_inputs(self, task_content, image):
        """
        Takes the standard 'task_content' dict and builds the
        LLaVA-specific conversation.
        """
        # 1. Get content from the standard dict
        full_prompt_text = task_content['full_prompt']
        has_image = task_content['has_image'] or task_content.get('has_blank_image', False) or task_content.get('has_gaussian_noise_image', False)
        
        # 2. Build the Qwen-VL 'conv' list
        conv = [{
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt_text},
            ],
        }]

        image_to_process = None
        if has_image:
            conv[0]['content'].append({"type": "image", "image": image})
        
        # 3. Apply chat template and tokenize - QwenVL Specific Stuff
        # [Refer --> https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct]

        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        image_inputs, _ = process_vision_info(conv)

        encoding = self.processor(
            text = [text],
            images = image_inputs,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        
        input_ids_list = encoding["input_ids"][0].tolist()
        return encoding, input_ids_list

    # get_chunk_ranges STAYS THE SAME. It already expects
    # a dict of strings ('task_chunk_strings')
    
    def load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config['model_id'],
            torch_dtype=getattr(torch, self.config['torch_dtype']),
            low_cpu_mem_usage=True,
            attn_implementation=self.config['attn_implementation'],
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.config['model_id'],use_fast=True, max_pixels=2500 * 28 * 28)
        # self.processor = AutoProcessor.from_pretrained(self.config['model_id'], max_pixels= 16 * 16 * 784, use_fast=True)
        # self.processor = AutoProcessor.from_pretrained(self.config['model_id'],use_fast=True, min_pixels=256 * 28 * 28, max_pixels=2500 * 28 * 28)
        return self.model, self.processor


    def get_special_token_ids(self):
        return {
            'image': 151655, # IMG_TOKEN_ID
            'eos': self.processor.tokenizer.eos_token_id
        }

    def get_chunk_ranges(self, input_ids_list, task_chunk_strings, image_token_id):


        all_chunks_ranges = {
            'im_start_0': [],
            'system': [],
            'post_system_newline': [],
            'system_instruction': [],
            'im_end_0': [],
            'post_im_end_0_newline': [],
            'im_start_1': [],
            'user': [],
            'post_user_newline': [],
            'text': [], 
            'instruction': [],
            'vision_start': [],
            'image': [],
            'vision_end': [],
            'im_end_1': [],
            'post_im_end_1_newline': [],
            'im_start_2': [],
            'assistant': [],
            'post_assistant_newline': [],
            'previously_generating_tokens':[],
            'currently_generating_token': []}

        
        tplt_im_start_str = "<|im_start|>"
        tplt_system_str = "system"
        tplt_newline_str = "\n"
        #TODO: This can't be made part of the task, since this is a model property. So fixing the system prompt here - may need to handle this better.
        # system_instruction_str = task_chunk_strings['system_instruction']
        system_instruction_str = "You are a helpful assistant."
        tplt_im_end_str = "<|im_end|>"
        #\n
        #im_start
        tplt_user_str = "user"
        #\n
        vision_start_str = "<|vision_start|>"
        # image_pad_str = "<|image_pad|>"
        vision_end_str = "<|vision_end|>"
        text_chunk_str = task_chunk_strings['text']
        instruction_chunk_str = task_chunk_strings['instruction']
        #im_end
        #\n
        #im_start
        tplt_assistant_str = "assistant"
        #\n
        #previous gen tokens
        #last gen token


        # QWEN STUFF GOES HERE
        im_start_ids = self.processor.tokenizer(tplt_im_start_str, add_special_tokens=False)["input_ids"]
        system_ids = self.processor.tokenizer(tplt_system_str, add_special_tokens=False)["input_ids"]
        newline_ids = self.processor.tokenizer(tplt_newline_str, add_special_tokens=False)["input_ids"]
        system_instruction_ids = self.processor.tokenizer(system_instruction_str, add_special_tokens=False)["input_ids"]
        im_end_ids = self.processor.tokenizer(tplt_im_end_str, add_special_tokens=False)["input_ids"]
        user_ids = self.processor.tokenizer(tplt_user_str, add_special_tokens=False)["input_ids"]
        vision_start_ids = self.processor.tokenizer(vision_start_str, add_special_tokens=False)["input_ids"]
        vision_end_ids = self.processor.tokenizer(vision_end_str, add_special_tokens=False)["input_ids"]
        text_ids = self.processor.tokenizer(text_chunk_str, add_special_tokens=False)["input_ids"]
        inst_ids = self.processor.tokenizer(instruction_chunk_str, add_special_tokens=False)["input_ids"]
        assistant_ids = self.processor.tokenizer(tplt_assistant_str, add_special_tokens=False)["input_ids"]


        for chunk in all_chunks_ranges.keys():
            occurence = 1

            if chunk == 'im_start_0':
                search_id = im_start_ids
            elif chunk == 'system':
                search_id = system_ids
            elif chunk == 'post_system_newline':
                search_id = newline_ids
            elif chunk == 'system_instruction':
                search_id = self.processor.tokenizer(system_instruction_str, add_special_tokens=False)["input_ids"]
            elif chunk == 'im_end_0':
                search_id = im_end_ids
            elif chunk == 'post_im_end_0_newline':
                search_id = newline_ids
                occurence = 2
            elif chunk == 'im_start_1':
                search_id = im_start_ids
                occurence = 2
            elif chunk == 'user':
                search_id = user_ids
            elif chunk == 'post_user_newline':
                search_id = newline_ids
                occurence = 3
            elif chunk == 'vision_start':
                search_id = vision_start_ids
            # elif chunk == 'image_pad':
            #     search_id = image_pad_str
            elif chunk == 'vision_end':
                search_id = vision_end_ids
            elif chunk == 'text':
                search_id = text_ids
            elif chunk == 'instruction':
                search_id = inst_ids
            elif chunk == 'im_end_1':
                search_id = im_end_ids
                occurence = 2
            elif chunk == 'post_im_end_1_newline':
                search_id = newline_ids
                occurence = 4
            elif chunk == 'im_start_2':
                search_id = im_start_ids
                occurence = 3
            elif chunk == 'assistant':
                search_id = assistant_ids
            elif chunk == 'post_assistant_newline':
                search_id = newline_ids
                occurence = 5
            else:
                continue

            start_idx = find_subsequence(input_ids_list, search_id, occurrence=occurence)
            assert start_idx is not None, f"Could not find {search_id} token {chunk}."
            end_idx = start_idx + len(search_id)
            all_chunks_ranges[chunk] = [start_idx, end_idx]
 
        # Find image tokens
        img_indices = [i for i, token_id in enumerate(input_ids_list) if token_id == image_token_id]
        if not img_indices:
            raise ValueError("Image token not found in input_ids")
        all_chunks_ranges['image'] = [img_indices[0], img_indices[-1] + 1]

        return all_chunks_ranges
    
    def get_tokenizer(self):
        """Returns the model's tokenizer object (for decoding, etc.)."""
        if not self.processor:
            raise ValueError("Processor not loaded. Call load_model() first.")
        return self.processor.tokenizer

    def get_attention_layers(self):
        """Returns the list of layer modules to register hooks on."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.model.language_model.layers

    def get_layer_indices(self):
        """Returns a list of layer indices (e.g., list(range(32)))."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        return list(range(self.model.config.text_config.num_hidden_layers))
    
    def get_num_heads(self):
        """Returns the number of heads for the model."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        # return self.model.config.text_config.num_attention_heads
        return self.model.config.text_config.num_key_value_heads