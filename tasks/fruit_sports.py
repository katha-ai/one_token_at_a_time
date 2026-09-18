from .base_task import BaseTask
from prompts.task_prompt_templates import prompt
class FruitSportsTask(BaseTask):
    
    def get_task_content(self, item):
        """
        Fetches 'sport', formats it, and returns the 
        standard content dictionary.
        """
        sport_text = item['sport_text']
        prompt_key = self.config['prompt_template']
        
        # 1. Get templates
        text_template = self.prompt_templates[prompt_key]['text_prompt']
        inst_template = self.prompt_templates[prompt_key]['instruction_prompt']

        # 2. Format chunks
        # These are the exact strings for chunk_strings
        text_chunk = text_template.replace("[[text_fill]]", f"{sport_text}")
        inst_chunk = inst_template # No fill-in for this example

        # 3. Create full prompt
        full_prompt_text = text_chunk + inst_chunk
        
        # 4. Return the standardized dictionary
        if self.config['image_type'] == 'blank':
            return {
                "full_prompt": full_prompt_text,
                "has_image": False,
                "has_blank_image": True,
                "chunks_for_attention": {
                    'text': text_chunk,
                    'instruction': inst_chunk
                }
            }
        else:
            return {
                "full_prompt": full_prompt_text,
                "has_image": True,
                "has_blank_image": False,
                "chunks_for_attention": {
                    'text': text_chunk,
                    'instruction': inst_chunk
                }
            }