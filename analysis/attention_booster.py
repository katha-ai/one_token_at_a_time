import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from functools import partial
import importlib

class BaseAttentionBooster:
    def apply_boost(self, input, original_softmax, dim, start_idx, end_idx, boost_factor, **s_kwargs):
        raise NotImplementedError

class LogitAttentionBooster(BaseAttentionBooster):
    def apply_boost(self, input, original_softmax, dim, start_idx, end_idx, boost_factor, **s_kwargs):
        # input shape (batch, heads, q_len, k_len)
        if input.ndim == 4:
            k_len = input.shape[-1]
            s_idx = max(0, start_idx)
            e_idx = min(k_len, end_idx)
            
            if s_idx < e_idx:
                # Multiply logits by factor
                modified_input = input.clone()
                modified_input[..., s_idx:e_idx] *= boost_factor
                return original_softmax(modified_input, dim=dim, **s_kwargs)
        
        return original_softmax(input, dim=dim, **s_kwargs)

class MultiplicativeAttentionBooster(BaseAttentionBooster):
    default_boost_factor = 1.0
    
    def apply_boost(self, input, original_softmax, dim, start_idx, end_idx, boost_factor, **s_kwargs):
        # input shape (batch, heads, q_len, k_len)
        if input.ndim == 4:
            k_len = input.shape[-1]
            s_idx = max(0, start_idx)
            e_idx = min(k_len, end_idx)
            
            if s_idx < e_idx:
                # 2. Safely compute baseline Softmax (handles all causal masks automatically)
                probs = original_softmax(input, dim=dim, **s_kwargs)
                
                # 3. Clone for modification
                modified_probs = probs.clone()
                
                # 4. CRITICAL FIX: Apply boost ONLY to the last query token `[-1:]`
                # This perfectly handles both use_cache=True (q_len=1) and use_cache=False (q_len=SeqLen)
                print(f"### Successfully applied boost of {boost_factor} to input of shape {input.shape}")
                modified_probs[..., -1:, s_idx:e_idx] *= boost_factor
                
                # 5. Renormalize (the clamp saves us from divide-by-zero if sum becomes 0)
                sum_probs = modified_probs.sum(dim=dim, keepdim=True).clamp(min=1e-12)
                return modified_probs / sum_probs
        
        print(f"### Returning ORIGINAL softmax of shape {input.shape}")
        return original_softmax(input, dim=dim, **s_kwargs)

class AttentionBooster:
    def __init__(self, model, boosting_config, chunk_ranges, special_token_ids):
        """
        Args:
            model: The HF model.
            boosting_config: Dict like {
                'boost_factor': 1.5, 
                'layers_to_boost': 'all', 
                'chunk': 'image',
                'step_to_boost': [0, 1, 2],
                'stop_on_token': '.',
                'boost_next_occurrence': False,
                'boost_type': 'logit' or 'multiplicative'
            }
            chunk_ranges: The dictionary mapping chunk names to [start, end] indices.
        """
        self.model = model
        self.config = boosting_config
        self.chunk_ranges = chunk_ranges
        self.special_token_ids = special_token_ids
        
        # Store original forward methods: {module_id: (module, original_fn)}
        self.original_forwards = {} 
        self.current_step = 0 

        # Global Switch
        self.is_active = True
        
        self.stop_token = self.config.get('stop_on_token', None)
        self.boost_next_occurrence = self.config.get('boost_next_occurrence', False)
        self.trigger_step = None 

        # Strategy selection
        boost_type = self.config.get('boost_type', 'logit')
        if boost_type == 'logit':
            self.strategy = LogitAttentionBooster()
        elif boost_type == 'multiplicative':
            self.strategy = MultiplicativeAttentionBooster()
        else:
            raise ValueError(f"Unknown boost_type: {boost_type}")

    def register_hooks(self):
        # boost_factor = self.config.get('boost_factor', 1.0)
        # if boost_factor == 1.0: 
        #     print("AttentionBooster: boost_factor is 1.0, skipping hook registration.")
        #     return

        layers_to_boost = self.config.get('layers_to_boost', 'all')
        
        # Determine which layers to hook
        all_layers = []
        if hasattr(self.model, 'layers'):
            all_layers = self.model.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            all_layers = self.model.model.layers
        elif hasattr(self.model, 'language_model'):
            all_layers = self.model.language_model.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            all_layers = self.model.model.language_model.layers
        else:
            raise ValueError("Could not locate model layers for boosting.")

        if layers_to_boost == 'all':
            target_layers = all_layers
        elif isinstance(layers_to_boost, list):
            target_layers = [all_layers[i] for i in layers_to_boost]
        else:
            raise ValueError(f"Invalid layers_to_boost: {layers_to_boost}")

        # Hybrid architectures (e.g. Qwen3.5-VL) mix softmax attention layers
        # (self_attn) with linear/SSM attention layers that have no self_attn
        # at all - boosting patches self_attn.forward's softmax, so those
        # layers must be skipped rather than AttributeError'ing.
        skipped = [layer for layer in target_layers if not hasattr(layer, 'self_attn')]
        target_layers = [layer for layer in target_layers if hasattr(layer, 'self_attn')]
        if skipped:
            print(f"AttentionBooster: skipping {len(skipped)} layer(s) without a softmax self_attn (e.g. linear/SSM attention layers).")

        for layer in target_layers:
            self_attn = layer.self_attn
            if id(self_attn) not in self.original_forwards:
                self.original_forwards[id(self_attn)] = (self_attn, self_attn.forward)
                self_attn.forward = types.MethodType(self._make_boosted_forward(self_attn), self_attn)
        
        print(f"AttentionBooster: Registered boosted forward for {len(target_layers)} layers using {self.strategy.__class__.__name__}.")

    def remove_hooks(self):
        for module_id, (module, original_fn) in self.original_forwards.items():
            module.forward = original_fn
        self.original_forwards = {}
        print("AttentionBooster: Removed all boosted forward hooks.")

    def update_step(self, step):
        self.current_step = step

    def check_and_update_state(self, generated_token_str):
        if not self.stop_token:
            return

        is_match = (generated_token_str == self.stop_token or generated_token_str.strip() == self.stop_token)

        if is_match:
            if self.boost_next_occurrence:
                if self.trigger_step is None: 
                    print(f"AttentionBooster: Trigger '{generated_token_str}' detected at step {self.current_step}. Boosting step {self.current_step + 1}.")
                    self.trigger_step = self.current_step + 1
                    self.is_active = True
            else:
                if self.is_active:
                    print(f"AttentionBooster: Stop token '{generated_token_str}' detected. Deactivating booster.")
                    self.is_active = False

    def _should_boost(self):
        if not self.is_active:
            return False

        step_list = self.config.get('step_to_boost')
        
        if step_list is not None:
            if isinstance(step_list, dict):
                return self.current_step in step_list
            if isinstance(step_list, list):
                return self.current_step in step_list
            return self.current_step == step_list
        
        if self.boost_next_occurrence:
            return self.trigger_step is not None and self.current_step == self.trigger_step
        
        return True

    def _get_target_chunk(self):
        step_list = self.config.get('step_to_boost')
        if isinstance(step_list, dict):
            return step_list.get(self.current_step)
        return self.config.get('chunk')

    def _make_boosted_forward(self, module):
        booster_self = self
        
        # We need to know where softmax is imported in the module of 'module'
        module_name = module.__class__.__module__
        target_mod = importlib.import_module(module_name)

        def boosted_forward(self, *args, **kwargs):
            # Check if we should boost
            if not booster_self._should_boost():
                # Call original forward
                _, original_fn = booster_self.original_forwards[id(self)]
                return original_fn(*args, **kwargs)

            # Target chunk info
            target_chunk = booster_self._get_target_chunk()
            if not target_chunk or target_chunk not in booster_self.chunk_ranges:
                # If chunk not found, just proceed normally
                _, original_fn = booster_self.original_forwards[id(self)]
                return original_fn(*args, **kwargs)

            start_idx, end_idx = booster_self.chunk_ranges[target_chunk]
            boost_factor = booster_self.config.get('boost_factor', 1.0)

            # Define patched softmax
            original_softmax = F.softmax
            
            def patched_softmax(input, dim=-1, **s_kwargs):
                # Calculate debug info safely
                if input.ndim == 4:
                    k_len = input.shape[-1]
                    s_idx = max(0, start_idx)
                    e_idx = min(k_len, end_idx)
                    
                    if s_idx < e_idx:
                        boost_type = booster_self.config.get('boost_type', 'logit')
                        
                        if boost_type == 'multiplicative':
                            # Get baseline mass
                            with torch.no_grad():
                                probs = original_softmax(input, dim=dim, **s_kwargs)
                                chunk_probs = probs[..., -1:, s_idx:e_idx]
                                current_mass = chunk_probs.sum().item() / (probs.shape[0] * probs.shape[1])
                                chunk_size = e_idx - s_idx
                                total_size = k_len
                            
                            # Perform the boost
                            res = booster_self.strategy.apply_boost(
                                input, original_softmax, dim, start_idx, end_idx, boost_factor, **s_kwargs
                            )
                            
                            # Get new mass
                            with torch.no_grad():
                                new_mass = res[..., -1:, s_idx:e_idx].sum().item() / (res.shape[0] * res.shape[1])
                                print(f"### [BOOST] Step {booster_self.current_step} | Factor {boost_factor} | Size {chunk_size}/{total_size} | Mass {current_mass:.4f} -> {new_mass:.4f}")
                            return res
                        else:
                            # Logit boosting
                            print(f"### [LOGIT-BOOST] Step {booster_self.current_step} | Size {e_idx-s_idx}/{k_len}")
                            return booster_self.strategy.apply_boost(
                                input, original_softmax, dim, start_idx, end_idx, boost_factor, **s_kwargs
                            )

                return booster_self.strategy.apply_boost(
                    input, original_softmax, dim, start_idx, end_idx, boost_factor, **s_kwargs
                )

            # Detect how softmax is accessed in the target module
            # If 'F' is used: target_mod.F.softmax
            # If 'softmax' is used directly: target_mod.softmax
            
            use_f = hasattr(target_mod, 'F') and hasattr(target_mod.F, 'softmax')
            use_direct = hasattr(target_mod, 'softmax')

            try:
                if use_f:
                    target_mod.F.softmax = patched_softmax
                if use_direct:
                    target_mod.softmax = patched_softmax
                
                # Also patch torch.nn.functional just in case
                F.softmax = patched_softmax
                
                _, original_fn = booster_self.original_forwards[id(self)]
                return original_fn(*args, **kwargs)
            finally:
                if use_f:
                    target_mod.F.softmax = original_softmax
                if use_direct:
                    target_mod.softmax = original_softmax
                F.softmax = original_softmax

        return boosted_forward
