import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from .base_handler import BaseModelHandler
from utils.token_utils import find_subsequence


class Qwen25OmniHandler(BaseModelHandler):
    """
    Model handler for Qwen2.5-Omni.

    Token sequence layout (audio + video):
        <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        <|im_start|>user\n
        <|audio_bos|><|AUDIO|>...<|AUDIO|><|audio_eos|>
        <|vision_bos|><|VIDEO|>...<|VIDEO|><|vision_eos|>
        {text}
        <|im_end|>\n
        <|im_start|>assistant\n
    """

    # ------------------------------------------------------------------ #
    #  Core interface                                                       #
    # ------------------------------------------------------------------ #

    def build_inputs(self, task_content, video, audio):
        """
        Build Qwen2.5-Omni-specific inputs from the standard task_content dict.
        Returns (encoding, input_ids_list).
        """
        full_prompt_text = task_content["full_prompt"]
        has_video = task_content["has_video"]
        has_audio = task_content["has_audio"]

        # Build the multimodal content list (order: audio → video → text)
        content = []
        if has_audio:
            content.append({"type": "audio", "audio": audio})
        if has_video:
            content.append({"type": "video", "video": video})
        content.append({"type": "text", "text": full_prompt_text})

        conversation = [{"role": "user", "content": content}]

        # apply_chat_template returns raw text for Qwen; processor handles
        # the actual tokenisation + multimodal feature extraction separately.
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        audios, images, videos = process_mm_info(
            conversation,
            use_audio_in_video=False,
        )

        encoding = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )

        input_ids_list = encoding["input_ids"][0].tolist()
        return encoding, input_ids_list

    # ------------------------------------------------------------------ #

    def load_model(self):
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.config["model_id"],
            torch_dtype=getattr(torch, self.config["torch_dtype"]),
            attn_implementation=self.config["attn_implementation"],
            device_map="auto",
        )
        # Disable the talker (audio-output) head — text-only inference
        self.model.disable_talker()
        self.model.eval()

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.config["model_id"])
        return self.model, self.processor

    # ------------------------------------------------------------------ #

    def get_special_token_ids(self):
        tok = self.processor.tokenizer
        return {
            "video":       tok.convert_tokens_to_ids("<|VIDEO|>"),
            "audio":       tok.convert_tokens_to_ids("<|AUDIO|>"),
            "eos":         tok.eos_token_id,
            # <|im_end|> closes every turn and is the natural "end-of-turn"
            "end_of_turn": tok.convert_tokens_to_ids("<|im_end|>"),
        }

    # ------------------------------------------------------------------ #

    def get_chunk_ranges(self, input_ids_list, task_chunk_strings,
                         video_token_id, audio_token_id):
        """
        Map every structural segment of the token sequence to a [start, end)
        index range.

        Newline occurrence accounting (4 newlines before generation prompt):
            1 → after 'system'
            2 → after first <|im_end|>     (post_system_newline)
            3 → after 'user'
            4 → after second <|im_end|>    (post_eot_newline)
            5 → after 'assistant'          (anchored, not searched)
        """
        all_chunks_ranges = {
            # ---- System turn ----
            "system_im_start":     [],
            "system":              [],
            "system_newline":      [],
            "system_content":      [],
            "system_im_end":       [],
            "post_system_newline": [],
            # ---- User turn header ----
            "user_im_start":       [],
            "user":                [],
            "user_newline":        [],
            # ---- Audio ----
            "audio_open":          [],   # <|audio_bos|>
            "audio":               [],   # <|AUDIO|> soft tokens
            "audio_close":         [],   # <|audio_eos|>
            # ---- Video ----
            "video_open":          [],   # <|vision_bos|>
            "video":               [],   # <|VIDEO|> soft tokens
            "video_close":         [],   # <|vision_eos|>
            # ---- Text ----
            "text":                [],
            "instruction":         [],
            # ---- End of user turn ----
            "end_of_turn":         [],
            "post_eot_newline":    [],
            # ---- Assistant turn prefix ----
            "assistant_im_start":  [],
            "assistant":           [],
            "post_assistant_newline": [],
            # ---- Generation ----
            "previously_generating_tokens": [],
            "currently_generating_token":   [],
        }

        tok = self.processor.tokenizer

        # ---- Tokenise every structural string ----
        def _ids(s):
            return tok(s, add_special_tokens=False)["input_ids"]

        im_start_ids    = _ids("<|im_start|>")
        im_end_ids      = _ids("<|im_end|>")
        system_ids      = _ids("system")
        system_msg_ids  = _ids("You are a helpful assistant.")
        user_ids        = _ids("user")
        assistant_ids   = _ids("assistant")
        newline_ids     = _ids("\n")
        audio_open_ids  = _ids("<|audio_bos|>")
        audio_close_ids = _ids("<|audio_eos|>")
        video_open_ids  = _ids("<|vision_bos|>")
        video_close_ids = _ids("<|vision_eos|>")
        text_ids        = _ids(task_chunk_strings["text"])
        inst_ids        = _ids(task_chunk_strings["instruction"])

        # ---- Sequential search (chunk → (token_ids, occurrence_index)) ----
        sequential_chunks = {
            # System turn
            "system_im_start":     (im_start_ids,    1),
            "system":              (system_ids,       1),
            "system_newline":      (newline_ids,      1),
            "system_content":      (system_msg_ids,   1),
            "system_im_end":       (im_end_ids,       1),
            "post_system_newline": (newline_ids,      2),
            # User turn
            "user_im_start":       (im_start_ids,     2),
            "user":                (user_ids,          1),
            "user_newline":        (newline_ids,       3),
            # Audio/video boundary tokens
            "audio_open":          (audio_open_ids,   1),
            "audio_close":         (audio_close_ids,  1),
            "video_open":          (video_open_ids,   1),
            "video_close":         (video_close_ids,  1),
            # Text payload
            "text":                (text_ids,         1),
            "instruction":         (inst_ids,         1),
            # End of user turn
            "end_of_turn":         (im_end_ids,       2),
            "post_eot_newline":    (newline_ids,       4),
            # Start of assistant turn (anchored below for 'assistant' and newline)
            "assistant_im_start":  (im_start_ids,     3),
        }

        for chunk, (search_ids, occurrence) in sequential_chunks.items():
            start_idx = find_subsequence(
                input_ids_list, search_ids, occurrence=occurrence
            )
            assert start_idx is not None, (
                f"Could not find '{chunk}' (occurrence={occurrence}) in input_ids."
            )
            all_chunks_ranges[chunk] = [start_idx, start_idx + len(search_ids)]

        # ---- 'assistant' and 'post_assistant_newline': anchor off im_start ----
        # (Avoids a false match if "assistant" appears in the prompt text)
        ast_start = all_chunks_ranges["assistant_im_start"][1]
        all_chunks_ranges["assistant"] = [
            ast_start,
            ast_start + len(assistant_ids),
        ]
        all_chunks_ranges["post_assistant_newline"] = [
            ast_start + len(assistant_ids),
            ast_start + len(assistant_ids) + len(newline_ids),
        ]

        # ---- Soft token spans (locate by token ID) ----
        aud_indices = [i for i, t in enumerate(input_ids_list) if t == audio_token_id]
        if aud_indices:
            all_chunks_ranges["audio"] = [aud_indices[0], aud_indices[-1] + 1]
        else:
            raise ValueError("Audio token not found in input_ids.")

        vid_indices = [i for i, t in enumerate(input_ids_list) if t == video_token_id]
        if vid_indices:
            all_chunks_ranges["video"] = [vid_indices[0], vid_indices[-1] + 1]
        else:
            raise ValueError("Video token not found in input_ids.")

        return all_chunks_ranges

    # ------------------------------------------------------------------ #
    #  Introspection helpers                                               #
    # ------------------------------------------------------------------ #

    def get_tokenizer(self):
        if not self.processor:
            raise ValueError("Processor not loaded. Call load_model() first.")
        return self.processor.tokenizer

    def get_attention_layers(self):
        """
        Qwen2.5-Omni uses a thinker-talker architecture; attention analysis
        targets the thinker's transformer layers.
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.thinker.model.layers

    def get_layer_indices(self):
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        # return list(range(self.model.thinker.config.num_hidden_layers))
        return list(range(self.model.thinker.config.text_config.num_hidden_layers))

    def get_num_heads(self):
        """Returns the number of heads for the model."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        # return self.model.thinker.config.num_key_value_heads
        return self.model.thinker.config.text_config.num_key_value_heads

    def get_generate_kwargs(self):
        """Returns model-specific kwargs for the generate() method."""
        return {
            "return_audio": False,
        }