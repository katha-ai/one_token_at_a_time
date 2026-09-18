import torch
import os
import numpy as np
import logging
from PIL import Image
from .attention_saver import AttentionSaver # Your class
from .valuenorm_saver import ValueNormSaver
from utils.generic_utils import get_filename_for_experiment
import json

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

    def process_sample(self, item, sample_idx, blocking_dict=None):
        # 1. Setup (Standard)
        base_name = get_filename_for_experiment(config=self.config,
                                            fruit_category=item.get('fruit_category', 'unknown_fruit'),
                                            sport_category=item.get('sport_category', 'unknown_sport'),
                                            sample_id=sample_idx,
                                            math_answer=item.get('math_answer', 'unknown_answer'),
                                            query_item=item.get('query_item', 'unknown_query'),
                                            image_count=item.get('image_count', 'unknown_image_count'),
                                            text_count=item.get('text_count', 'unknown_text_count'),
                                            label=item.get('label'))
        
        task_content = self.task.get_task_content(item)

        # ... (Input building logic remains the same) ...
        if self.model.config.name_or_path in ['Qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2-0.5B-Instruct']:
             encoding, input_ids_list = self.model_handler.build_inputs(task_content)
        else:
            if task_content['has_image']:
                fname = os.path.join(self.config['data']['image_base_dir'], item['image_path'])
            elif task_content['has_blank_image']:
                fname = self.config['data']['black_image_path']
            elif task_content['has_gaussian_noise_image']:
                fname = self.config['data']['gaussian_noise_image_path']
            else:
                raise ValueError("Task content must specify either 'has_image' or 'has_blank_image'.")
            encoding, input_ids_list = self.model_handler.build_inputs(task_content, fname)
        
        image_token_id = self.special_token_ids.get('image', None)
        all_chunks_ranges = self.model_handler.get_chunk_ranges(input_ids_list, task_content['chunks_for_attention'], image_token_id)
        
        device = self.model.device
        inputs = {k: v.to(device) for k, v in encoding.items()}
        _ = inputs.pop("offset_mapping", None)

        # --- Capture Input Tokens ---
        input_tokens_ids = inputs["input_ids"][0]
        input_tokens_list = [self.tokenizer.decode(tid, skip_special_tokens=False) for tid in input_tokens_ids]

        # --- Initialize the 4-Bucket Storage ---
        # These are the lists that will hold the averaged time-series
        series_storage = {
            "Image": [],
            "Text": [],
            "Instruction": [],
            "Previous": []
        }
        
        generated_tokens_list = []

        # --- Helper: Map dynamic chunk names to your 4 buckets ---
        def get_bucket_name(chunk_key):
            k = chunk_key.lower()
            if "image" in k: return "Image"
            if "previously_generating" in k: return "Previous"
            if "instruction" in k or "system" in k: return "Instruction"
            # Fallback: assume everything else (question, query, etc.) is "Text"
            if "text" in k: return "Text"

        # ... (Blocker setup remains same) ...
        blocker = None
        if blocking_dict:
            from .attention_blocker import AttentionBlocker 
            blocker = AttentionBlocker(self.model, blocking_dict, all_chunks_ranges, self.special_token_ids)

        try:
            # 3. Generation Loop
            for step in range(1, self.config['generation']['num_tokens_to_generate']+1):
                seq_len = inputs["input_ids"].shape[1]
                
                if step == 1:
                    prev_start = prev_end = None
                else:
                    prev_start = seq_len - step
                    prev_end = seq_len - 1
                
                all_chunks_ranges['previously_generating_tokens'] = [prev_start, prev_end]
                all_chunks_ranges['currently_generating_token'] = [seq_len - 1, seq_len]
        
                attn_saver = AttentionSaver(all_chunks_ranges)
                handles = []
                
                try:
                    for layer in self.attention_layers:
                        handles.append(layer.self_attn.register_forward_hook(attn_saver))
                        
                    if blocker:
                        blocker.update_step(step)
                        blocker.chunk_ranges = all_chunks_ranges 
                        blocker.register_hooks()

                    with torch.no_grad():
                        output = self.model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=False)
                finally:
                    for handle in handles: handle.remove()
                    if blocker: blocker.remove_hooks()
                
                new_token_id = output[:, -1:]
                decoded_str = self.tokenizer.decode(new_token_id[0], skip_special_tokens=True)
                if blocker: blocker.check_and_update_state(decoded_str)
                print(f"Decoded: '{decoded_str}' at step {step} | sample {base_name}")

                if new_token_id.item() == self.special_token_ids['eos']:
                    break

                new_token_id = new_token_id.to(inputs['input_ids'].device).long()
                generated_tokens_list.append(decoded_str)

                # --- NEW: Aggregate Across Layers for this Step ---
                captured_attentions = attn_saver.get_captured_attentions()
                
                if not captured_attentions:
                    break

                # Temporary accumulators for this specific step
                step_totals = {"Image": 0.0, "Text": 0.0, "Instruction": 0.0, "Previous": 0.0}
                step_counts = {"Image": 0, "Text": 0, "Instruction": 0, "Previous": 0}

                # Iterate over selected layers
                layers_processed = 0
                for layer_idx in self.layers_to_save:
                    if layer_idx >= len(captured_attentions): continue
                    
                    layers_processed += 1
                    layer_data = captured_attentions[layer_idx]
                    target_prefix = "currently_generating_token__attends_to__"

                    for causal_key, attn_val in layer_data.items():
                        if causal_key.startswith(target_prefix):
                            # Extract raw chunk name (e.g. "image_chunk")
                            raw_chunk_name = causal_key.replace(target_prefix, "")
                            
                            # Map to your 4 buckets
                            if raw_chunk_name not in ['image', 'text', 'instruction', 'previously_generating_tokens']:
                                continue
                            bucket = get_bucket_name(raw_chunk_name)
                            
                            # attn_val can be None if a chunk (like 'Previous') is empty in step 1.
                            if attn_val is None:
                                val = 0.0
                            else:
                                val = float(attn_val)
                            
                            # Add to step total
                            step_totals[bucket] += val
                            step_counts[bucket] += 1

                # Calculate Average for this step and append to Series
                # Note: We divide by layers_processed to get the average attention *per layer*
                if layers_processed > 0:
                    for bucket in series_storage:
                        # If a bucket wasn't found in this step (e.g. no "Previous" in step 1), store 0.0
                        # We divide by layers_processed because we summed up the attention from N layers.
                        # Note: If multiple chunks mapped to the same bucket (rare), this logic sums them. 
                        # If you want pure average, divide by layers_processed.
                        
                        avg_val = step_totals[bucket] / layers_processed
                        series_storage[bucket].append(avg_val)
                # --------------------------------------------------

                inputs["input_ids"] = torch.cat([inputs["input_ids"], new_token_id], dim=1)
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones((new_token_id.shape[0],1), device=new_token_id.device, dtype=inputs["attention_mask"].dtype)], dim=1)
                
                del attn_saver, captured_attentions
                torch.cuda.empty_cache()

            # --- Save Single JSON ---
            print(f"Saving averaged attention JSON for sample {base_name}...")
            
            output_dict = {
                "sample_id": base_name,
                "input_tokens": input_tokens_list,
                "output_tokens": generated_tokens_list,
                "series": series_storage
            }

            # Save to "averaged_attention" folder
            out_dir = os.path.join(self.config['save_dir'], "averaged_attention")
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"{base_name}.json")
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(output_dict, f, indent=4)

            self.logger.info(f"Successfully processed and saved sample {base_name}.")

        finally:
            del inputs, encoding, task_content
            if blocker: del blocker
            torch.cuda.empty_cache()