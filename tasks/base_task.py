from abc import ABC, abstractmethod

class BaseTask(ABC):
    def __init__(self, task_config, prompt_templates):
        self.config = task_config
        self.prompt_templates = prompt_templates # Your central dict

    @abstractmethod
    def get_task_content(self, item):
        """
        Takes a data row 'item' and returns a standardized 
        dictionary containing:
        - 'full_prompt': The final text for the model.
        - 'has_image': A boolean.
        - 'has_blank_image': A boolean (either has_image or a blank image, if both, blank will be taken).
        - 'chunks_for_attention': A dict of strings to find (e.g., {'text': ..., 'instruction': ...})
        """
        pass