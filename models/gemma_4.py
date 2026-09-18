import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from .base_handler import BaseModelHandler
from utils.token_utils import find_subsequence
from PIL import Image


class Gemma4Handler(BaseModelHandler):
    """Handler for google/gemma-4-12B-it (Gemma4UnifiedForConditionalGeneration),
    the image-capable sibling of the video/audio-only google/gemma-4-E4B-it model
    already supported by models/gemma_4_omni.py. Same underlying architecture/
    processor class, different checkpoint and modality (image, not video/audio).
    """

    def build_inputs(self, task_content, image):
        """
        Takes the standard 'task_content' dict and builds the Gemma-4 (unified)
        conversation. image: path of the image file.
        """
        full_prompt_text = task_content['full_prompt']
        has_image = task_content['has_image'] or task_content.get('has_blank_image', False) or task_content.get('has_gaussian_noise_image', False)

        content = []
        if has_image:
            image = Image.open(image).convert("RGB")
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": full_prompt_text})

        conv = [{
            "role": "user",
            "content": content,
        }]

        # enable_thinking=False: without it, the model-turn prefix is left
        # open-ended and the model free-generates its own
        # `<|channel>thought...<channel|>` reasoning block before the actual
        # answer, eating into num_tokens_to_generate for no benefit here.
        # With it, the template itself appends a pre-closed, empty thought
        # channel as part of the fixed prefix (see get_chunk_ranges below -
        # channel_open/thought/post_thought_newline/channel_close), so
        # generation starts directly on the answer. Matches how
        # models/gemma_4_omni.py already calls this same processor family.
        encoding = self.processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        # Work around a bug in this preview transformers build
        # (Gemma4UnifiedForConditionalGeneration): the image processor always
        # right-pads pixel_values/image_position_ids to max_soft_tokens (280),
        # marking padding rows with image_position_ids == (-1,-1). The model
        # is supposed to strip those before scattering vision features into
        # the text embedding sequence, and does so correctly when
        # get_image_features is called standalone - but through the normal
        # forward path (model.generate() or a direct model(**inputs) call,
        # both under device_map="auto") this intermittently fails to apply,
        # scattering all 280 features (14 pure padding) into the 266 real
        # image-token slots and raising "Image features and image tokens do
        # not match, tokens: 266, features: 280". Trimming here removes the
        # padding before the model ever sees it, so there's nothing left for
        # that codepath to mishandle.
        if "image_position_ids" in encoding:
            image_position_ids = encoding["image_position_ids"]
            valid = (image_position_ids[0] != -1).all(dim=-1)
            encoding["pixel_values"] = encoding["pixel_values"][:, valid, :]
            encoding["image_position_ids"] = image_position_ids[:, valid, :]

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
            'image': 258880,  # <|image_soft_token|>
            'eos': self.processor.tokenizer.eos_token_id,
            'end_of_turn': 106,  # <turn|>
        }

    def get_chunk_ranges(self, input_ids_list, task_chunk_strings, image_token_id):

        all_chunks_ranges = {
            # --- Template structure ---
            'bos': [],
            'start_of_turn_1': [],
            'user': [],
            'newline': [],
            # --- Image ---
            'image_open': [],             # <|image> (boi)
            'image': [],                  # <|image|> soft tokens
            'image_close': [],            # <image|> (eoi)
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
            # --- Pre-closed empty thought channel (enable_thinking=False) ---
            'channel_open': [],
            'thought': [],
            'post_thought_newline': [],
            'channel_close': [],
            'previously_generating_tokens': [],
            'currently_generating_token': []
        }

        # --- Tokenize template strings ---
        tplt_bos           = "<bos>"
        tplt_start_of_turn = "<|turn>"
        tplt_user          = "user"
        tplt_newline       = "\n"
        tplt_end_of_turn   = "<turn|>"
        tplt_model         = "model"
        tplt_channel_open  = "<|channel>"
        tplt_thought       = "thought"
        tplt_channel_close = "<channel|>"
        text_chunk_str     = task_chunk_strings['text'].lstrip()
        instruction_str    = task_chunk_strings['instruction']

        tok = self.processor.tokenizer
        bos_ids          = tok(tplt_bos,           add_special_tokens=False)["input_ids"]
        sot_ids          = tok(tplt_start_of_turn, add_special_tokens=False)["input_ids"]
        user_ids         = tok(tplt_user,          add_special_tokens=False)["input_ids"]
        newline_ids      = tok(tplt_newline,       add_special_tokens=False)["input_ids"]
        text_ids         = tok(text_chunk_str,     add_special_tokens=False)["input_ids"]
        inst_ids         = tok(instruction_str,    add_special_tokens=False)["input_ids"]
        eot_ids          = tok(tplt_end_of_turn,   add_special_tokens=False)["input_ids"]
        model_ids        = tok(tplt_model,         add_special_tokens=False)["input_ids"]
        channel_open_ids = tok(tplt_channel_open,  add_special_tokens=False)["input_ids"]
        thought_ids      = tok(tplt_thought,       add_special_tokens=False)["input_ids"]
        channel_close_ids = tok(tplt_channel_close, add_special_tokens=False)["input_ids"]

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

        # --- 'model' onward: anchor off start_of_turn_2, not independent search ---
        # (avoids false match if e.g. the word 'model' or 'thought' appears in the question text)
        sot2_end = all_chunks_ranges['start_of_turn_2'][1]
        all_chunks_ranges['model'] = [sot2_end, sot2_end + len(model_ids)]
        model_end = sot2_end + len(model_ids)
        all_chunks_ranges['post_model_newline'] = [model_end, model_end + len(newline_ids)]
        pmn_end = model_end + len(newline_ids)
        all_chunks_ranges['channel_open'] = [pmn_end, pmn_end + len(channel_open_ids)]
        co_end = pmn_end + len(channel_open_ids)
        all_chunks_ranges['thought'] = [co_end, co_end + len(thought_ids)]
        th_end = co_end + len(thought_ids)
        all_chunks_ranges['post_thought_newline'] = [th_end, th_end + len(newline_ids)]
        ptn_end = th_end + len(newline_ids)
        all_chunks_ranges['channel_close'] = [ptn_end, ptn_end + len(channel_close_ids)]

        # --- Image soft tokens ---
        img_indices = [i for i, t in enumerate(input_ids_list) if t == image_token_id]
        if not img_indices:
            raise ValueError("Image token not found in input_ids.")
        all_chunks_ranges['image']       = [img_indices[0],     img_indices[-1] + 1]
        all_chunks_ranges['image_open']  = [img_indices[0] - 1, img_indices[0]]
        all_chunks_ranges['image_close'] = [img_indices[-1] + 1, img_indices[-1] + 2]

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
        return self.model.config.text_config.num_key_value_heads
