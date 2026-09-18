import torch
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from .base_handler import BaseModelHandler
from utils.token_utils import find_subsequence


class Qwen35VLHandler(BaseModelHandler):
    """
    Model handler for Qwen3.5-VL (e.g. Qwen/Qwen3.5-9B).

    Two structural differences from Qwen2.5-VL (models/qwen_25_vl.py) drive
    everything below:

    1. Chat template. Qwen3.5's default template has NO system turn (unlike
       Qwen2.5-VL's hardcoded "You are a helpful assistant." block), and
       always emits a `<think>...</think>` block right after
       `<|im_start|>assistant\n`. Passing `enable_thinking=False` to
       apply_chat_template collapses that block to a fixed, empty
       `<think>\n\n</think>\n\n` (verified against the actual tokenizer -
       see qwen35vl_fruit_math_inference.py, the throwaway harness this
       handler formalizes). Token layout (content order is image-then-text,
       also matching that harness - NOT the text-then-image order
       Qwen2.5-VL's handler uses):

           <|im_start|>user\n
           <|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>
           {text}{instruction}
           <|im_end|>\n
           <|im_start|>assistant\n<think>\n\n</think>\n\n

    2. Attention architecture. Qwen3.5's text backbone is a HYBRID of linear
       (SSM/gated-delta-net, `layer.linear_attn`) and full softmax
       (`layer.self_attn`) attention layers - only every 4th layer
       (`full_attention_interval: 4`) is a standard softmax layer; the rest
       have no `self_attn` attribute at all. This framework's attention
       extraction/boosting hooks are self_attn-based (they inspect softmax
       attention weights / patch F.softmax inside self_attn.forward), so
       get_attention_layers() below returns ONLY the full-attention layers -
       hooking a linear_attention layer's (nonexistent) self_attn would
       AttributeError, and even if it existed, gated-delta-net has no
       chunk-to-chunk softmax attention map to extract.

       Consequently the layer_idx used for saving (0..7) is a POSITION in
       this filtered list, not the underlying model's decoder layer index.
       self.full_attention_model_layer_indices records the mapping back to
       real model layer indices for reference.

       NOTE: blocking (analysis/attention_blocker.py) hooks the whole
       decoder layer's attention_mask, which linear_attention layers receive
       in a completely different (non-additive-float) format built by
       create_recurrent_attention_mask - blocking has NOT been validated
       against this model and should be treated as unsupported until that's
       checked; boosting (self_attn-only) is fine as long as
       `layers_to_boost` only ever names full-attention layers (see
       attention_booster.py's filtering of layers lacking self_attn).
    """

    def build_inputs(self, task_content, image):
        """
        image: path of the image file on disk.
        """
        full_prompt_text = task_content['full_prompt']
        has_image = task_content['has_image'] or task_content.get('has_blank_image', False) or task_content.get('has_gaussian_noise_image', False)

        content = []
        if has_image:
            raw_image = Image.open(image).convert("RGB")
            content.append({"type": "image", "image": raw_image})
        content.append({"type": "text", "text": full_prompt_text})

        conv = [{
            "role": "user",
            "content": content,
        }]

        encoding = self.processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids_list = encoding["input_ids"][0].tolist()
        return encoding, input_ids_list

    def load_model(self):
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.config['model_id'],
            torch_dtype=getattr(torch, self.config['torch_dtype']),
            low_cpu_mem_usage=True,
            attn_implementation=self.config['attn_implementation'],
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.config['model_id'], use_fast=True, max_pixels=2500 * 28 * 28)

        layer_types = self.model.config.text_config.layer_types
        self.full_attention_model_layer_indices = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]

        return self.model, self.processor

    def get_special_token_ids(self):
        tok = self.processor.tokenizer
        return {
            'image': self.model.config.image_token_id,
            'vision_start': self.model.config.vision_start_token_id,
            'vision_end': self.model.config.vision_end_token_id,
            'eos': tok.eos_token_id,
        }

    def get_chunk_ranges(self, input_ids_list, task_chunk_strings, image_token_id):

        all_chunks_ranges = {
            'user_im_start': [],
            'user': [],
            'post_user_newline': [],
            'vision_start': [],
            'image': [],
            'vision_end': [],
            'text': [],
            'instruction': [],
            'end_of_turn': [],
            'post_eot_newline': [],
            'assistant_im_start': [],
            'assistant': [],
            'post_assistant_newline': [],
            'think_start': [],
            'post_think_open_newline': [],
            'think_end': [],
            'post_think_close_newline': [],
            'previously_generating_tokens': [],
            'currently_generating_token': []}

        tok = self.processor.tokenizer

        def _ids(s):
            return tok(s, add_special_tokens=False)["input_ids"]

        im_start_ids = _ids("<|im_start|>")
        im_end_ids = _ids("<|im_end|>")
        user_ids = _ids("user")
        assistant_ids = _ids("assistant")
        newline_ids = _ids("\n")
        vision_start_ids = _ids("<|vision_start|>")
        vision_end_ids = _ids("<|vision_end|>")
        text_ids = _ids(task_chunk_strings['text'])
        inst_ids = _ids(task_chunk_strings['instruction'])
        think_start_ids = _ids("<think>")
        think_end_ids = _ids("</think>")
        think_newline_ids = _ids("\n\n")

        sequential_chunks = {
            'user_im_start': (im_start_ids, 1),
            'user': (user_ids, 1),
            'post_user_newline': (newline_ids, 1),
            'vision_start': (vision_start_ids, 1),
            'vision_end': (vision_end_ids, 1),
            'text': (text_ids, 1),
            'instruction': (inst_ids, 1),
            'end_of_turn': (im_end_ids, 1),
            'post_eot_newline': (newline_ids, 2),
            'assistant_im_start': (im_start_ids, 2),
        }

        for chunk, (search_ids, occurrence) in sequential_chunks.items():
            start_idx = find_subsequence(input_ids_list, search_ids, occurrence=occurrence)
            assert start_idx is not None, f"Could not find '{chunk}' (occurrence={occurrence}) in input_ids."
            all_chunks_ranges[chunk] = [start_idx, start_idx + len(search_ids)]

        # 'assistant' onward is a fixed, deterministic suffix emitted by the
        # chat template (enable_thinking=False) - anchor off assistant_im_start
        # rather than re-searching, both to avoid false matches of these common
        # words/tokens in the prompt text and because it's simply guaranteed.
        cursor = all_chunks_ranges['assistant_im_start'][1]

        for chunk, search_ids in [
            ('assistant', assistant_ids),
            ('post_assistant_newline', newline_ids),
            ('think_start', think_start_ids),
            ('post_think_open_newline', think_newline_ids),
            ('think_end', think_end_ids),
            ('post_think_close_newline', think_newline_ids),
        ]:
            assert input_ids_list[cursor:cursor + len(search_ids)] == search_ids, (
                f"Expected '{chunk}' tokens {search_ids} at position {cursor}, "
                f"found {input_ids_list[cursor:cursor + len(search_ids)]}. "
                "The fixed post-assistant template (enable_thinking=False) may have changed."
            )
            all_chunks_ranges[chunk] = [cursor, cursor + len(search_ids)]
            cursor += len(search_ids)

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
        """Returns only the full (softmax) attention decoder layers - see class docstring."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        layer_types = self.model.config.text_config.layer_types
        all_layers = self.model.model.language_model.layers
        return [layer for layer, lt in zip(all_layers, layer_types) if lt == "full_attention"]

    def get_layer_indices(self):
        """
        Positions within get_attention_layers()'s filtered list (0..7), NOT
        real model decoder-layer indices - see self.full_attention_model_layer_indices.
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        return list(range(len(self.get_attention_layers())))

    def get_num_heads(self):
        """Returns the number of heads for the model."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.config.text_config.num_key_value_heads
