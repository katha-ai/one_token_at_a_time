import torch

class AttentionSaver:
    def __init__(self, chunk_ranges):
        """
        Initializes the saver with the chunk ranges for on-the-fly processing.
        """
        self.attentions = []  # This will now store list[dict], not list[tensor]
        self.chunk_ranges = chunk_ranges
        self.chunk_keys = list(chunk_ranges.keys())

    def __call__(self, module, inputs, outputs):
        """
        A forward hook function to SLICE, PROCESS, and SAVE attention scalars.
        """
        if not (isinstance(outputs, tuple) and len(outputs) > 1):
            return

        # 1. Get the attention tensor (e.g., in bfloat16) - KEEP ON GPU
        attn_tensor = outputs[1].detach()

        # 2. This dict will store all processed scalars for *this layer*
        layer_processed_attentions = {}

        # 3. Perform the slicing and aggregation that was in your main loop
        for src_chunk_idx in range(len(self.chunk_keys)):
            for tgt_chunk_idx in range(src_chunk_idx, len(self.chunk_keys)):
                
                src_chunk_name = self.chunk_keys[src_chunk_idx]
                tgt_chunk_name = self.chunk_keys[tgt_chunk_idx]
                causal_chunk_name = f"{tgt_chunk_name}__attends_to__{src_chunk_name}"

                src_start, src_end = self.chunk_ranges[src_chunk_name]
                tgt_start, tgt_end = self.chunk_ranges[tgt_chunk_name]

                # Check for None, which happens on the first step for 'previously_generating_tokens'
                if src_start is None or tgt_start is None:
                    attn_value = None
                else:
                    # 4. Slice the tensor *on the GPU*
                    # Handle use_cache=True where tgt dimension is 1 (the new token)
                    # and seq_len dimension is total sequence length.
                    # attn_tensor shape: [batch, heads, tgt_len, src_len]
                    
                    actual_tgt_len = attn_tensor.shape[2]
                    actual_src_len = attn_tensor.shape[3]
                    
                    # If we are in decoding mode (tgt_len=1), we can only attend from the current token.
                    # We check if the requested tgt_range overlaps with the tokens present in this forward pass.
                    # In our loop, we only care about the last token attending to others during decoding.
                    
                    if actual_tgt_len == 1:
                        # We are in decoding. The only 'tgt' token is the last one.
                        # Its global index is actual_src_len - 1.
                        # We only compute if the requested tgt_chunk includes this last token.
                        global_tgt_idx = actual_src_len - 1
                        if tgt_start <= global_tgt_idx < tgt_end:
                            # The single token in this tensor is within the requested target chunk
                            attn_block = attn_tensor[:, :, 0:1, src_start:src_end]
                            attn_value = attn_block.sum(-1).mean(-1).mean().item()
                        else:
                            attn_value = 0.0
                    else:
                        # We are in prefill or use_cache=False mode. Full matrix available.
                        attn_block = attn_tensor[:, :, tgt_start:tgt_end, src_start:src_end]
                        attn_value = attn_block.sum(-1).mean(-1).mean().item()

                layer_processed_attentions[causal_chunk_name] = attn_value
        
        # 6. Append the *small dictionary* of scalars, not the 1.2GB tensor
        self.attentions.append(layer_processed_attentions)

    def clear(self):
        """Clears the stored attention dictionaries."""
        self.attentions = []

    def get_captured_attentions(self):
        """Returns the list of processed attention dictionaries."""
        return self.attentions