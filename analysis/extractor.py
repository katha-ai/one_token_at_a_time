import torch
import os
import numpy as np
import logging
from PIL import Image
from .attention_saver import AttentionSaver # Your class
from .valuenorm_saver import ValueNormSaver
from utils.generic_utils import get_filename_for_experiment
import glob

class AttentionExtractor:
    def __init__(self, model_handler, task, config):
        self.model_handler = model_handler
        self.task = task
        self.config = config
        self.logger = logging.getLogger(__name__)

        # --- REFACTORED SECTION ---
        # Don't grab internals. Ask the handler for components.
        self.model = model_handler.model
        
        # This line is the key fix for your question.
        # It gets self.processor.tokenizer from a VLM
        # and self.tokenizer from an LLM.
        self.tokenizer = model_handler.get_tokenizer()
        
        # This fixes the .language_model.layers problem
        self.attention_layers = model_handler.get_attention_layers()
        
        # This fixes the .config.text_config problem
        self.layers_to_save = model_handler.get_layer_indices()
        
        self.special_token_ids = model_handler.get_special_token_ids()

        self.num_heads = model_handler.get_num_heads()
        # --- END REFACTORED SECTION ---

    def process_sample(self, item, sample_idx, blocking_dict=None, boosting_dict=None):
        # 1. Prepare inputs
        
        base_name = get_filename_for_experiment(config=self.config,
                                            fruit_category = item.get('fruit_category', 'unknown_fruit'),
                                            sport_category = item.get('sport_category', 'unknown_sport'),
                                            sample_id = sample_idx,
                                            math_answer = item.get('math_answer', 'unknown_answer'),
                                            query_item = item.get('query_item', 'unknown_query'),
                                            image_count = item.get('image_count', 'unknown_image_count'),
                                            text_count = item.get('text_count', 'unknown_text_count'),
                                            label=item.get('label'),
                                            mma_original_id = item.get('original_id', 'unknown_mma_id'),
                                            mma_answer = item.get('answer', 'unknown_mma_answer'),
                                            mms_class_label = item.get('class_label', 'unknown_class'))
        
        
        task_content = self.task.get_task_content(item)

        if self.model.config.name_or_path in ['Qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2-0.5B-Instruct']:
            # 3. Get inputs and chunk ranges from the Model Handler
            encoding, input_ids_list = self.model_handler.build_inputs(task_content)
            
            # Get 'image' token ID, default to None if not present (for LLMs)
            image_token_id = self.special_token_ids.get('image', None)
            
            all_chunks_ranges = self.model_handler.get_chunk_ranges(
                input_ids_list,
                task_content['chunks_for_attention'], 
                image_token_id
            )
        elif self.model.config.name_or_path in ['google/gemma-4-E4B-it']:
            sample_id = item['original_id']
            vid_name = item['video_id']
            vid_st = item['vid_start']
            vid_end = item['vid_end']
            base = f"{sample_id}__video_{vid_name}__{vid_st}_{vid_end}"
            # matches = glob.glob(os.path.join(self.config['data']['video_base_dir'], f"*{base}*"))
            # if not matches:
            #     raise FileNotFoundError(f"No video file found matching base '{base}' in directory {self.config['data']['video_base_dir']}")
            matches = glob.glob(os.path.join(self.config['data']['video_base_dir'], f"{base}*"))
            if not matches:
                print(f"[SKIP] No video found for {base}")
                return None
            elif len(matches) > 1:
                raise ValueError(f"Multiple video files found matching base '{base}' in directory {self.config['data']['video_base_dir']}: {matches}")

            vid_path = matches[0]

            aud_name = item['audio_id']
            aud_st = item['audio_start']
            aud_end = item['audio_end']
            base_aud = f"{sample_id}__audio_{aud_name}__{aud_st}_{aud_end}"
            matches_aud = glob.glob(os.path.join(self.config['data']['audio_base_dir'], f"{base_aud}*"))
            matches_aud = glob.glob(os.path.join(self.config['data']['audio_base_dir'], f"{base_aud}*"))
            if not matches_aud:
                print(f"[SKIP] No audio found for {base_aud}")
                return None
            elif len(matches_aud) > 1:
                raise ValueError(f"Multiple audio files found matching base '{base_aud}' in directory {self.config['data']['audio_base_dir']}: {matches_aud}")
            
            aud_path = matches_aud[0]

            encoding, input_ids_list = self.model_handler.build_inputs(task_content, vid_path, aud_path)

            all_chunks_ranges = self.model_handler.get_chunk_ranges(
                input_ids_list,
                task_content['chunks_for_attention'], 
                self.special_token_ids.get('video', None),
                self.special_token_ids.get('audio', None)
            )

        else:

            # Only load image if the task requires it
            if task_content['has_image']:
                fname = os.path.join(self.config['data']['image_base_dir'], item['image_path'])
            
            elif task_content['has_blank_image']:
                fname = self.config['data']['black_image_path']
                # raw_image = Image.new('RGB', (448, 448), color = 'black')
            elif task_content['has_gaussian_noise_image']:
                fname = self.config['data']['gaussian_noise_image_path']
            else:
                raise ValueError("Task content must specify either 'has_image' or 'has_blank_image'.")
            
            encoding, input_ids_list = self.model_handler.build_inputs(task_content, fname)

             # Get 'image' token ID, default to None if not present (for LLMs)
            image_token_id = self.special_token_ids.get('image', None)
            
            all_chunks_ranges = self.model_handler.get_chunk_ranges(
                input_ids_list,
                task_content['chunks_for_attention'], 
                image_token_id
            )
            
        
       
        
        device = self.model.device
        inputs = {k: v.to(device) for k, v in encoding.items()}
        _ = inputs.pop("offset_mapping", None)

        # DEBUG: Check input keys and shapes
        # print(f"DEBUG: Initial input keys: {list(inputs.keys())}")
        # for k, v in inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"DEBUG: Input {k} shape: {v.shape}")
        #     else:
        #         print(f"DEBUG: Input {k} type: {type(v)}")


        # --- BLOCKING SETUP ---
        blocker = None
        if blocking_dict:
            from .attention_blocker import AttentionBlocker # Import the class above #TODO:::CHANGE TO OG for the 'GOOD BLOCKING'
            # from .attention_blocker_leakage import AttentionBlocker # Import the class above #TODO:::CHANGE TO OG for the 'GOOD BLOCKING'
            blocker = AttentionBlocker(
                model=self.model, 
                blocking_config=blocking_dict,
                chunk_ranges=all_chunks_ranges,
                special_token_ids=self.special_token_ids
            )
        # ----------------------

        # --- BOOSTING SETUP ---
        booster = None
        boost_step_metadata = {}
        if boosting_dict:
            from .attention_booster import AttentionBooster
            booster = AttentionBooster(
                model=self.model,
                boosting_config=boosting_dict,
                chunk_ranges=all_chunks_ranges,
                special_token_ids=self.special_token_ids
            )
            # Record which step(s) got boosted per chunk, saved into every
            # layer's .npz below as `boosted_step__{chunk}` - so which step
            # was actually targeted (e.g. for `random` / `precomputed` /
            # `precomputed_partial_random` styles, where it varies per
            # sample) doesn't have to be re-derived from the run config
            # after the fact.
            step_to_boost = boosting_dict.get('step_to_boost')
            fixed_chunk = boosting_dict.get('chunk')
            if isinstance(step_to_boost, dict):
                # {step: chunk} - multi-chunk styles (precomputed/random/
                # precomputed_partial_random with a list chunk_to_boost)
                chunk_to_steps = {}
                for step, chunk in step_to_boost.items():
                    if chunk is None:
                        continue
                    chunk_to_steps.setdefault(chunk, []).append(step)
                for chunk, steps in chunk_to_steps.items():
                    boost_step_metadata[f"boosted_step__{chunk}"] = np.array(sorted(steps))
            elif isinstance(step_to_boost, list) and fixed_chunk:
                boost_step_metadata[f"boosted_step__{fixed_chunk}"] = np.array(sorted(step_to_boost))
            elif isinstance(step_to_boost, (int, float)) and fixed_chunk:
                boost_step_metadata[f"boosted_step__{fixed_chunk}"] = np.array([int(step_to_boost)])
        # ----------------------

        attention_progression = {}
        for src_chunk_idx in range(len(all_chunks_ranges)):
            for tgt_chunk_idx in range(src_chunk_idx, len(all_chunks_ranges)):
                src_chunk_name = list(all_chunks_ranges.keys())[src_chunk_idx]
                tgt_chunk_name = list(all_chunks_ranges.keys())[tgt_chunk_idx]
                causal_chunk_name = f"{tgt_chunk_name}__attends_to__{src_chunk_name}"
                for layer_idx in self.layers_to_save:
                    if layer_idx not in attention_progression:
                        attention_progression[layer_idx] = {}
                    attention_progression[layer_idx][causal_chunk_name] = []
    

        generated_tokens_list = []
        try:
            # 3. Start the step-by-step generation loop
            for step in range(1, self.config['generation']['num_tokens_to_generate']+1):
                seq_len = inputs["input_ids"].shape[1]
                if step == 1:
                    prev_generated_tokens_start_idx = prev_generated_tokens_end_idx = None
                else:
                    prev_generated_tokens_start_idx = seq_len - step
                    prev_generated_tokens_end_idx = seq_len - 1
                
                all_chunks_ranges['previously_generating_tokens'] = [prev_generated_tokens_start_idx, prev_generated_tokens_end_idx]
                all_chunks_ranges['currently_generating_token'] = [seq_len - 1, seq_len]
        
                attn_saver = AttentionSaver(all_chunks_ranges)
                # val_saver = ValueNormSaver(all_chunks_ranges, self.num_heads)
                handles = []
                # Use the abstracted self.attention_layers
                try:
                    for layer in self.attention_layers:

                        handle1 = layer.self_attn.register_forward_hook(attn_saver)
                        handles.append(handle1)
                        
                    # 2. Blocking Hooks (New code)
                    if blocker:
                        blocker.update_step(step)
                        # Re-update chunk ranges in blocker because seq_len increased
                        blocker.chunk_ranges = all_chunks_ranges 
                        blocker.register_hooks()
                    
                    # 3. Boosting Hooks (New code)
                    if booster:
                        booster.update_step(step)
                        booster.chunk_ranges = all_chunks_ranges
                        booster.register_hooks()
                    # ----------------------
                    with torch.no_grad():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=1,
                            do_sample=False,
                            use_cache=False,
                        )
                finally:
                    for handle in handles:
                        handle.remove()
                    
                    if blocker:
                        blocker.remove_hooks()
                    
                    if booster:
                        booster.remove_hooks()
                
                new_token_id = output[:, -1:]
                
                # --- KEY FIX ---
                # Use self.tokenizer directly
                decoded_str = self.tokenizer.decode(new_token_id[0], skip_special_tokens=True)


                #--- Update Blocker State ---
                if blocker:
                    blocker.check_and_update_state(decoded_str)

                #--- Update Booster State ---
                if booster:
                    booster.check_and_update_state(decoded_str)


                print(f"New token ID: {new_token_id} | Decoded: '{decoded_str}' at step {step} | sample {base_name}")
                # --- END KEY FIX ---

                if new_token_id.item() == self.special_token_ids['eos'] or new_token_id.item() == self.special_token_ids.get('end_of_turn', -1):
                    print(f"Encountered EOS token at step {step} for sample {base_name}. Stopping generation.")
                    break

                new_token_id = new_token_id.to(inputs['input_ids'].device).long()

                # --- KEY FIX ---
                decoded_token = self.tokenizer.decode(new_token_id[0], skip_special_tokens=True)
                # --- END KEY FIX ---
                generated_tokens_list.append(decoded_token)

                captured_attentions = attn_saver.get_captured_attentions()
                # captured_norms = val_saver.get_captured_norms()
                if not captured_attentions:
                    print(f"Warning: Hook failed on step {step} for sample {sample_idx}.")
                    break
                    
                # Use the abstracted self.layers_to_save
                for layer_idx in self.layers_to_save:
                    # Check if hook captured data for this layer (it might not if layer_idx is out of range)
                    if layer_idx < len(captured_attentions):
                        layer_data_dict = captured_attentions[layer_idx]
                        for causal_chunk_name, attn_value in layer_data_dict.items():
                            if causal_chunk_name in attention_progression[layer_idx]:
                                # If attn_value is None (happens on first step for empty chunks), store 0.0
                                val = 0.0 if attn_value is None else float(attn_value)
                                attention_progression[layer_idx][causal_chunk_name].append(val)
                            else:
                                # This can happen if a chunk was None in the first step
                                self.logger.warning(f"Mismatch in chunk keys: {causal_chunk_name} not initialized.")
                    else:
                        self.logger.warning(f"Layer index {layer_idx} out of range for captured attentions (len {len(captured_attentions)}).")


                inputs["input_ids"] = torch.cat([inputs["input_ids"], new_token_id], dim=1)
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones((new_token_id.shape[0],1), device=new_token_id.device, dtype=inputs["attention_mask"].dtype)], dim=1)
                
                if "token_type_ids" in inputs:
                    inputs["token_type_ids"] = torch.cat([inputs["token_type_ids"], torch.zeros((new_token_id.shape[0],1), device=new_token_id.device, dtype=inputs["token_type_ids"].dtype)], dim=1)
                
                if "mm_token_type_ids" in inputs:
                    inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros((new_token_id.shape[0],1), device=new_token_id.device, dtype=inputs["mm_token_type_ids"].dtype)], dim=1)
                
                # DEBUG: Check input shapes after update
                # print(f"DEBUG Step {step} update: input_ids={inputs['input_ids'].shape}, attention_mask={inputs.get('attention_mask', torch.tensor([])).shape}")
                # for k, v in inputs.items():
                #     if k not in ["input_ids", "attention_mask"] and isinstance(v, torch.Tensor):
                #         print(f"DEBUG Step {step} other input: {k} shape={v.shape}")
                
                del attn_saver, captured_attentions
                torch.cuda.empty_cache()
            
            # ... (Your saving logic is correct) ...
            print(f"Saving attention progression for sample {base_name}...")
            for layer_idx, progression_dict in attention_progression.items():
                if not progression_dict or not any(progression_dict.values()):
                    print(f"Warning: No attention data captured for layer {layer_idx}, sample {base_name}. Skipping save.")
                    continue

                progression_out_dir = os.path.join(self.config['save_dir'], str(layer_idx), "attention_progression")
                os.makedirs(progression_out_dir, exist_ok=True)
                save_path = os.path.join(progression_out_dir, base_name)
                progression_dict['generated_tokens'] = generated_tokens_list
                if boost_step_metadata:
                    progression_dict.update(boost_step_metadata)
                np.savez_compressed(save_path, **progression_dict)

            self.logger.info(f"Successfully processed and saved sample {base_name}.")
        finally:
            del inputs, encoding, attention_progression, task_content
            if blocker:
                del blocker
            torch.cuda.empty_cache()
