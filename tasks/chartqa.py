from .base_task import BaseTask

class ChartQATask(BaseTask):
    def get_task_content(self, item):
        """
        ChartQA task content with two questions.
        """
        q1 = item['q1']
        q2 = item['q2']
        prompt_key = self.config['prompt_template']
        
        # 1. Get templates
        text_template = self.prompt_templates[prompt_key]['text_prompt']
        inst_template = self.prompt_templates[prompt_key]['instruction_prompt']

        # 2. Format chunks
        text_chunk = text_template.replace("[[q1]]", q1).replace("[[q2]]", q2)
        inst_chunk = inst_template

        # 3. Create full prompt
        full_prompt_text = text_chunk + inst_chunk
        
        # 4. Return the standardized dictionary
        return {
            "full_prompt": full_prompt_text,
            "has_image": True,
            "has_blank_image": False,
            "chunks_for_attention": {
                'text': text_chunk,
                'instruction': inst_chunk
            }
        }
