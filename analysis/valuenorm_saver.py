import torch

class ValueNormSaver:
    def __init__(self, chunk_ranges, num_heads):
        """
        Args:
            chunk_ranges: Dict of {name: [start, end]}
            num_heads: int, required to reshape the linear layer output correctly.
        """
        self.value_norms = []
        self.chunk_ranges = chunk_ranges
        self.chunk_keys = list(chunk_ranges.keys())
        self.num_heads = num_heads

    def __call__(self, module, inputs, outputs):
        """
        Hooks into the Linear layer (v_proj). 
        Outputs of v_proj are usually [Batch, Seq, Hidden_Dim].
        """
        # outputs is the projected Value tensor
        v_tensor = outputs.detach() 
        
        # 1. Reshape to separate heads: [Batch, Seq, Num_Heads, Head_Dim]
        # We infer Head_Dim dynamically
        b, s, hidden_dim = v_tensor.shape
        head_dim = hidden_dim // self.num_heads
        
        # Reshape: [Batch, Seq, Heads, Head_Dim]
        v_tensor = v_tensor.view(b, s, self.num_heads, head_dim)

        # 2. Calculate Norm per head: [Batch, Seq, Heads]
        # We use float32 for norm calculation to avoid overflow/underflow in bfloat16
        norms_per_head = torch.norm(v_tensor.float(), p=2, dim=-1)

        # 3. Average across heads to get 'Token Signal Strength': [Batch, Seq]
        # (You could also keep heads separate, but to match your scalar setup, we mean)
        token_norms = norms_per_head.mean(dim=-1)

        # 4. Dictionary for this layer
        layer_processed_norms = {}

        for chunk_name in self.chunk_keys:
            start, end = self.chunk_ranges[chunk_name]

            if start is None:
                norm_value = None
            else:
                # Slice the sequence dimension [Batch, START:END]
                chunk_block = token_norms[:, start:end]
                
                # Average magnitude of tokens in this chunk
                norm_value = chunk_block.mean().item()

            layer_processed_norms[chunk_name] = norm_value

        self.value_norms.append(layer_processed_norms)

    def clear(self):
        self.value_norms = []

    def get_captured_norms(self):
        return self.value_norms