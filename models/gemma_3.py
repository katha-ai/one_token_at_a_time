import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from .base_handler import BaseModelHandler
from utils.token_utils import find_subsequence
from PIL import Image
class GemmaHandler(BaseModelHandler):

    # In class LlavaOnevisionHandler(BaseModelHandler):

    def build_inputs(self, task_content, image):
        """
        Takes the standard 'task_content' dict and builds the
        Gemma3-specific conversation.
        image: Path of the image file.
        """
        # 1. Get content from the standard dict
        full_prompt_text = task_content['full_prompt']
        has_image = task_content['has_image'] or task_content.get('has_blank_image', False) or task_content.get('has_gaussian_noise_image', False)
        
        # 2. Build the LLaVA-specific 'conv' list
        content = []
        image_to_process = None
        # content.append({"type": "text", "text": full_prompt_text})
        if has_image:
            try:
                image = Image.open(image).convert("RGB")
                content.append({"type": "image", 'image': image})
            except Exception as e:
                self.logger.warning(f"Could not open image {image}: {e}. Skipping.")
                return

            image_to_process = image
        content.append({"type": "text", "text": full_prompt_text})

        conv = [{
            "role": "user",
            "content": content,
        }]

        # 3. Apply chat template and tokenize
        # prompt_str = self.processor.apply_chat_template(conv, add_generation_prompt=True)
        # encoding = self.processor(
        #     text=prompt_str,
        #     images=image_to_process,
        #     return_tensors="pt",
        #     return_offsets_mapping=True,
        # )

        encoding = self.processor.apply_chat_template(
                                                        conv,
                                                        add_generation_prompt=True,
                                                        tokenize=True,
                                                        return_dict=True,
                                                        return_tensors="pt",
                                                    )
        input_ids_list = encoding["input_ids"][0].tolist()
        return encoding, input_ids_list

    # get_chunk_ranges STAYS THE SAME. It already expects
    # a dict of strings ('task_chunk_strings')
    
    def load_model(self):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.config['model_id'],
            torch_dtype=getattr(torch, self.config['torch_dtype']),
            low_cpu_mem_usage=True,
            attn_implementation=self.config['attn_implementation'],
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.config['model_id'], use_fast=True)
        return self.model, self.processor


    def get_special_token_ids(self):
        return {
            'image': 262144, # <|image_soft_token|>
            'eos': self.processor.tokenizer.eos_token_id
        }

    def get_chunk_ranges(self, input_ids_list, task_chunk_strings, image_token_id):
                
        all_chunks_ranges = {
            'bos': [],
            'start_of_turn_1': [],
            'user': [],
            'newline_group_3': [],
            'start_of_image': [],
            'image': [],
            'end_of_image': [],
            'newline_group_2': [],
            'text': [],
            'instruction': [],
            'end_of_turn': [],
            'post_eot_newline': [],
            'start_of_turn_2': [],
            'model': [],
            'post_model_newline': [],
            'previously_generating_tokens': [],
            'currently_generating_token': []
        }
        #TODO: The prompt, instruction should be made dynamic based on the task - each model can run with different prompt and instructions.
        # Tokenize chunk strings
        
        tplt_bos = "<bos>"
        tplt_start_of_turn = "<start_of_turn>"
        tplt_user_str = "user"
        tplt_newline_group_3_str = "\n\n\n"
        tplt_start_of_image = "<start_of_image>"
        tplt_end_of_image = "<end_of_image>"
        tplt_newline_group_2_str = "\n\n"
        text_chunk_str = task_chunk_strings['text'].lstrip() # Remove leading/trailing whitespace to avoid tokenization issues
        instruction_chunk_str = task_chunk_strings['instruction']
        tplt_end_of_turn = "<end_of_turn>"
        tplt_newline_str = "\n"
        #tplt_start_of_turn
        tplt_model_str = "model"
        #\n
        #previous gen tokens
        #last gen token

        bos_ids = self.processor.tokenizer(tplt_bos, add_special_tokens=False)["input_ids"]
        start_of_turn_ids = self.processor.tokenizer(tplt_start_of_turn, add_special_tokens=False)["input_ids"]
        user_ids = self.processor.tokenizer(tplt_user_str, add_special_tokens=False)["input_ids"]
        newline_g3_ids = self.processor.tokenizer(tplt_newline_group_3_str, add_special_tokens=False)["input_ids"]
        start_of_image_ids = self.processor.tokenizer(tplt_start_of_image, add_special_tokens=False)["input_ids"]
        end_of_image_ids = self.processor.tokenizer(tplt_end_of_image, add_special_tokens=False)["input_ids"]
        newline_g2_ids = self.processor.tokenizer(tplt_newline_group_2_str, add_special_tokens=False)["input_ids"]
        text_ids = self.processor.tokenizer(text_chunk_str, add_special_tokens=False)["input_ids"]
        inst_ids = self.processor.tokenizer(instruction_chunk_str, add_special_tokens=False)["input_ids"]
        end_of_turn_ids = self.processor.tokenizer(tplt_end_of_turn, add_special_tokens=False)["input_ids"]
        newline_ids = self.processor.tokenizer(tplt_newline_str, add_special_tokens=False)["input_ids"]
        model_ids = self.processor.tokenizer(tplt_model_str, add_special_tokens=False)["input_ids"]

        for chunk in all_chunks_ranges.keys():
            occurence = 1
            if chunk == 'bos':
                search_id = bos_ids
            elif chunk == 'start_of_turn_1':
                search_id = start_of_turn_ids
            elif chunk == 'user':
                search_id = user_ids
            elif chunk == 'newline_group_3':
                search_id = newline_g3_ids
            elif chunk == 'start_of_image':
                search_id = start_of_image_ids
            elif chunk == 'end_of_image':
                search_id = end_of_image_ids
            elif chunk == 'newline_group_2':
                search_id = newline_g2_ids
            elif chunk == 'text':
                search_id = text_ids
            elif chunk == 'instruction':
                search_id = inst_ids
            elif chunk == 'end_of_turn':
                search_id = end_of_turn_ids
            elif chunk == 'post_eot_newline':
                search_id = newline_ids
            elif chunk == 'start_of_turn_2':
                search_id = start_of_turn_ids
                occurence = 2
            elif chunk == 'model':
                search_id = model_ids
            elif chunk == 'post_model_newline':
                search_id = newline_ids
                occurence = 2
            else:
                continue

            start_idx = find_subsequence(input_ids_list, search_id, occurrence=occurence)
            assert start_idx is not None, f"Could not find token "+chunk+" in input_ids."
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
        return self.model.language_model.layers

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