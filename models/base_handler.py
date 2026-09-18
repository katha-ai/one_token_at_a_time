from abc import ABC, abstractmethod

class BaseModelHandler(ABC):
    def __init__(self, model_config):
        self.config = model_config
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        """
        Loading the model and processor into self.model and self.processor.
        """
        pass

    # @abstractmethod
    # def get_inputs_and_ids(self, conv, image):
    #     """
    #     Process text and image, return tokenized 'inputs' and 'input_id_list'.
    #     """
    #     pass

    @abstractmethod
    def build_inputs(self, task_content, image):
        """
        Takes the standard 'task_content' dict and builds model-specific inputs.
        Returns tokenized 'inputs' and 'input_id_list'.
        """
        pass

    @abstractmethod
    def get_chunk_ranges(self, input_ids_list, task_chunk_strings, image_token_id):
        """
        Takes the token list and text chunks returns the all_chunk_ranges dict.
        all_chunk_ranges: {
            'chunk': [start_idx, end_idx]
            }
        """
        pass

    @abstractmethod
    def get_special_token_ids(self):
        """
        Returns a dict of special token ids. {'image': image_token_id, ...}
        """
        pass

    
    @abstractmethod
    def get_tokenizer(self):
        """Returns the model's tokenizer object (for decoding, etc.)."""
        pass

    @abstractmethod
    def get_attention_layers(self):
        """Returns the list of layer modules to register hooks on."""
        pass

    @abstractmethod
    def get_layer_indices(self):
        """Returns a list of layer indices (e.g., list(range(32)))."""
        pass

    @abstractmethod
    def get_num_heads(self):
        """Returns the number of heads for the model."""
        pass

    def get_generate_kwargs(self):
        """Returns a dict of model-specific kwargs for generate()."""
        return {}