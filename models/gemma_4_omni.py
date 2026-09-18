import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from .base_handler import BaseModelHandler
from utils.token_utils import find_subsequence
from PIL import Image

class Gemma4OmniHandler(BaseModelHandler):

    # In class LlavaOnevisionHandler(BaseModelHandler):

    def build_inputs(self, task_content, video, audio):
        """
        Takes the standard 'task_content' dict and builds the
        Gemma4-specific conversation.
        """
        # 1. Get content from the standard dict
        full_prompt_text = task_content['full_prompt']
        
        has_video = task_content['has_video']
        has_audio = task_content['has_audio']

        # 2. Build the LLaVA-specific 'conv' list
        content = []
        
        
        if has_audio:
            content.append({"type": "audio", 'audio': audio})

        if has_video:
            content.append({"type": "video", 'video': video})

        content.append({"type": "text", "text": full_prompt_text})

        conv = [{
            "role": "user",
            "content": content,
        }]

        encoding = self.processor.apply_chat_template(
                                                        conv,
                                                        add_generation_prompt=True,
                                                        tokenize=True,
                                                        return_dict=True,
                                                        return_tensors="pt",
                                                        enable_thinking=False
                                                    )
        input_ids_list = encoding["input_ids"][0].tolist()
        return encoding, input_ids_list
    
    def load_model(self):
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.config['model_id'],
            torch_dtype=getattr(torch, self.config['torch_dtype']),
            low_cpu_mem_usage=True,
            attn_implementation=self.config['attn_implementation'],
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.config['model_id'], use_fast=True, padding_side="left")
        return self.model, self.processor

    def get_special_token_ids(self):
        return {
            'video': 258884, # <|image_soft_token|>
            'audio': 258881,
            'eos': self.processor.tokenizer.eos_token_id,
            'end_of_turn': 106
        }

    
    def get_chunk_ranges(self, input_ids_list, task_chunk_strings, video_token_id, audio_token_id):

        all_chunks_ranges = {
            # --- Template structure ---
            'bos': [],
            'start_of_turn_1': [],
            'user': [],
            'newline': [],
            # --- Video ---
            'video_header': [],           # first timestamp + <|image> open tag
            'video': [],                  # <|video|> soft tokens (spans inter-frame structure)
            'video_close': [],            # <image|> after last frame
            # --- Audio ---
            'audio_open': [],             # <|audio>
            'audio': [],                  # <|audio|> soft tokens
            'audio_close': [],            # <audio|>
            # --- Text ---
            'text': [],
            'instruction': [],
            # --- End of user turn ---
            'end_of_turn': [],
            'post_eot_newline': [],
            # --- Model turn prefix ---
            'start_of_turn_2': [],
            'model': [],
            'post_model_newline': [],
            'previously_generating_tokens':[],
            'currently_generating_token': []
        }

        # --- Tokenize template strings ---
        tplt_bos           = "<bos>"
        tplt_start_of_turn = "<|turn>"
        tplt_user          = "user"
        tplt_model         = "model"
        tplt_newline       = "\n"
        tplt_end_of_turn   = "<turn|>"
        text_chunk_str     = task_chunk_strings['text'].lstrip()
        instruction_str    = task_chunk_strings['instruction']

        tok = self.processor.tokenizer
        bos_ids          = tok(tplt_bos,           add_special_tokens=False)["input_ids"]
        sot_ids          = tok(tplt_start_of_turn, add_special_tokens=False)["input_ids"]
        user_ids         = tok(tplt_user,          add_special_tokens=False)["input_ids"]
        model_ids        = tok(tplt_model,         add_special_tokens=False)["input_ids"]
        newline_ids      = tok(tplt_newline,       add_special_tokens=False)["input_ids"]
        text_ids         = tok(text_chunk_str,     add_special_tokens=False)["input_ids"]
        inst_ids         = tok(instruction_str,    add_special_tokens=False)["input_ids"]
        eot_ids          = tok(tplt_end_of_turn,   add_special_tokens=False)["input_ids"]

        # --- Chunks found via find_subsequence ---
        sequential_chunks = {
            'bos':              (bos_ids,     1),
            'start_of_turn_1':  (sot_ids,     1),
            'user':             (user_ids,    1),
            'newline':          (newline_ids, 1),
            'text':             (text_ids,    1),
            'instruction':      (inst_ids,    1),
            'end_of_turn':      (eot_ids,     1),
            'post_eot_newline': (newline_ids, 2),
            'start_of_turn_2':  (sot_ids,     2),
        }

        for chunk, (search_ids, occurrence) in sequential_chunks.items():
            start_idx = find_subsequence(input_ids_list, search_ids, occurrence=occurrence)
            assert start_idx is not None, f"Could not find '{chunk}' in input_ids."
            all_chunks_ranges[chunk] = [start_idx, start_idx + len(search_ids)]

        # --- 'model' and 'post_model_newline': anchor off start_of_turn_2, not independent search ---
        # (avoids false match if the word 'model' appears in the question text)
        sot2_end = all_chunks_ranges['start_of_turn_2'][1]
        all_chunks_ranges['model'] = [sot2_end, sot2_end + len(model_ids)]
        all_chunks_ranges['post_model_newline'] = [sot2_end + len(model_ids),
                                                sot2_end + len(model_ids) + len(newline_ids)]

        # --- Video soft tokens ---
        vid_indices = [i for i, t in enumerate(input_ids_list) if t == video_token_id]
        if not vid_indices:
            raise ValueError("Video token not found in input_ids.")
        all_chunks_ranges['video']       = [vid_indices[0],     vid_indices[-1] + 1]
        # video_header: everything between \n and first soft token (timestamp + <|image>)
        all_chunks_ranges['video_header'] = [all_chunks_ranges['newline'][1], vid_indices[0]]
        # video_close: one token immediately after last soft token (<image|>)
        all_chunks_ranges['video_close'] = [vid_indices[-1] + 1, vid_indices[-1] + 2]

        # --- Audio soft tokens ---
        aud_indices = [i for i, t in enumerate(input_ids_list) if t == audio_token_id]
        if not aud_indices:
            raise ValueError("Audio token not found in input_ids.")
        all_chunks_ranges['audio']       = [aud_indices[0],     aud_indices[-1] + 1]
        # audio_open: one token immediately before first soft token (<|audio>)
        all_chunks_ranges['audio_open']  = [aud_indices[0] - 1, aud_indices[0]]
        # audio_close: one token immediately after last soft token (<audio|>)
        all_chunks_ranges['audio_close'] = [aud_indices[-1] + 1, aud_indices[-1] + 2]

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