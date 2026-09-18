from math import e
import torch

class AttentionBlocker:
    def __init__(self, model, blocking_config, chunk_ranges, special_token_ids):
        """
        Args:
            model: The HF model.
            blocking_config: Dict like {'block_type': 'layer_collapsed', 'step_to_block': [...], ...}
            chunk_ranges: The dictionary mapping chunk names to [start, end] indices.
        """
        self.model = model
        self.config = blocking_config
        self.chunk_ranges = chunk_ranges
        self.special_token_ids = special_token_ids
        self.handles = []
        self.current_step = 0 

        # Global Switch
        self.is_active = True
        
        self.stop_token = self.config.get('stop_on_token', None)
        
        # New Flag: Defaults to False to keep original behavior intact
        self.block_next_occurrence = self.config.get('block_next_occurrence', False)
        
        # State tracker for the new mode
        self.trigger_step = None 

    def register_hooks(self):
        block_type = self.config.get('block_type', None)
        if not block_type: return

        layers_to_hook = []
        if block_type == 'layer_collapsed':
            if hasattr(self.model, 'layers'):
                layers_to_hook = self.model.layers
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                layers_to_hook = self.model.model.layers
            elif hasattr(self.model, 'language_model'):
                layers_to_hook = self.model.language_model.layers
            elif hasattr(self.model.model, 'language_model'):
                layers_to_hook = self.model.model.language_model.layers
            else:
                raise ValueError("Could not locate model layers for blocking.")
        
        for layer in layers_to_hook:
            handle = layer.register_forward_pre_hook(self._blocking_hook_fn, with_kwargs=True)
            self.handles.append(handle)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def update_step(self, step):
        self.current_step = step

    def check_and_update_state(self, generated_token_str):
        """
        Updates state based on generated tokens.
        """
        if not self.stop_token:
            return

        # Check for token match
        is_match = (generated_token_str == self.stop_token or generated_token_str.strip() == self.stop_token)

        if is_match:
            if self.block_next_occurrence:
                # MODE 3: Dynamic (Next Only)
                # CHECK: Have we already triggered? If so, ignore subsequent dots.
                if self.trigger_step is None: 
                    print(f"Trigger '{generated_token_str}' detected at step {self.current_step}. Blocking step {self.current_step + 1}.")
                    self.trigger_step = self.current_step + 1
                    self.is_active = True
            else:
                # MODE 2: Dynamic (Until)
                if self.is_active: # Only print if we are actually changing state
                    print(f"Dynamic Block: Stop token '{generated_token_str}' detected. Deactivating blocker.")
                    self.is_active = False

    def _blocking_hook_fn(self, module, args, kwargs):
        if not self.is_active:
            return args, kwargs

        # ------------------------------------------------------------------
        # 1. DECISION LOGIC: SHOULD WE BLOCK THIS STEP?
        # ------------------------------------------------------------------
        
        step_list = self.config.get('step_to_block')
        should_block = False

        if step_list is not None:
            # === MODE 1: PRECOMPUTED (Static List) ===
            # Strict adherence to the list. 
            if isinstance(step_list, list):
                if self.current_step in step_list:
                    should_block = True
            elif self.current_step == step_list:
                should_block = True
        
        else:
            # === DYNAMIC MODES (No step_to_block defined) ===
            if self.block_next_occurrence:
                # === MODE 3: DYNAMIC (Next Only) ===
                # Block ONLY if we are at the specific triggered step
                if self.trigger_step is not None and self.current_step == self.trigger_step:
                    should_block = True
            else:
                # === MODE 2: DYNAMIC (Until) ===
                # We block continuously until check_and_update_state sets is_active=False
                # (Since we are here, is_active is True, so we block)
                should_block = True

        # If the decision logic says "Don't block", we exit immediately
        if not should_block:
            return args, kwargs

        # ------------------------------------------------------------------
        # 2. BLOCKING MECHANISM (Your original code)
        # ------------------------------------------------------------------
        
        target_chunk_name = self.config.get('chunk')
        if target_chunk_name not in self.chunk_ranges:
            print(f"WARNING: Chunk '{target_chunk_name}' not found.")
            return args, kwargs
        
        start_idx, end_idx = self.chunk_ranges[target_chunk_name]
        
        mask_tensor = kwargs.get('attention_mask')
        is_in_kwargs = True
        
        if mask_tensor is None and len(args) > 1:
            mask_tensor = args[1]
            is_in_kwargs = False

        if mask_tensor is None:
            return args, kwargs 

        modified_mask = mask_tensor.clone()
        min_dtype = torch.finfo(modified_mask.dtype).min
        #st 1000  end 2000 --> keys: 2000 to end query 1000 to 2000
        # 
        
        if end_idx < modified_mask.shape[-1]:
            if self.config['blocking_method'] == 'lazy':
                modified_mask[..., -1:, :end_idx] = min_dtype
            elif self.config['blocking_method'] == 'total':
                modified_mask[..., end_idx:, start_idx:end_idx] = min_dtype
            else:
                raise ValueError(f"Unknown blocking method: {self.config['blocking_method']}")
        if is_in_kwargs:
            kwargs['attention_mask'] = modified_mask
            return args, kwargs
        else:
            new_args = list(args)
            new_args[1] = modified_mask
            return tuple(new_args), kwargs