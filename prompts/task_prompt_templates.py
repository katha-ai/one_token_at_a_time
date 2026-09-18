prompt = {
    'fruit_sport_prompt': {
        "text_prompt": " 'Text': [[text_fill]]",
        "instruction_prompt": " First, identify the fruit in the image and then identify the sport from the provided 'Text'. Generate response in the format: The fruit in the image is [fruit_name]. The sport in the text is [sport]."
    },
    'fruit_math_prompt': {
        "text_prompt": " 'Text': [[text_fill]]",
        "instruction_prompt": " First, identify the fruit in the image provided and then answer the math puzzle in 'Text'. Generate response in the format: The fruit in the image is [fruit_name]. The answer to the math puzzle is [numeric_answer]."
    },
    'math_fruit_prompt': {
        "text_prompt": " 'Text': [[text_fill]]",
        "instruction_prompt": " First, answer the math puzzle in 'Text' and then identify the fruit in the image provided. Generate response in the format: The answer to the math puzzle is [numeric_answer]. The fruit in the image is [fruit_name]."
    },
    'vsr_prompt_vanilla': {
        "text_prompt": " 'Text': [[caption]]",
        "instruction_prompt": " Answer the following spatial relation: [[instruction]]. Generate response in the format: [object_1] is [spatial_relation] [object_2]"
    },

    'vsr_prompt_text': {
        "text_prompt": " 'Text': [[caption]]",
        "instruction_prompt": " Answer the following spatial relation based on 'Text': [[instruction]]. Generate response in the format: [object_1] is [spatial_relation] [object_2]"
    },

    'vsr_prompt_image': {
        "text_prompt": " 'Text': [[caption]]",
        "instruction_prompt": " Answer the following spatial relation based on the Image provided: [[instruction]]. Generate response in the format: [object_1] is [spatial_relation] [object_2]"
    },

    'vsr_prompt_image_hard': {
        "text_prompt": " 'Text': [[caption]]",
        "instruction_prompt": " Answer the following spatial relation solely based on the Image provided: [[instruction]]. Generate response in the format: As per the image [object_1] is [spatial_relation] [object_2]"
    },

    'vsr_prompt_discrepancy': {
        "text_prompt": " 'Text': [[caption]]",
        "instruction_prompt": " There exists a discrepancy between the spatial location of key objects conveyed in the 'Text' compared to the Image provided. Identify this spatial discrepancy and answer in the format: the image has: [spatial_relation] while the text mentions: [spatial_relation]."
    },

    'vsr_prompt_discrepancy_updated': {
        "text_prompt": " 'Text': [[caption]]",
        "instruction_prompt": " There exists a discrepancy between the spatial location of key objects conveyed in the 'Text' compared to the Image provided: [[instruction]]. Identify this spatial discrepancy and answer in the format: the image has [object 1] [spatial_relation] [object 2] while the text mentions [object 1] [spatial_relation] [object 2]."
    },

    'chartqa_prompt': {
        "text_prompt": " 'Text': 'Q1': [[q1]], 'Q2': [[q2]]",
        "instruction_prompt": " First solve Q1 and then solve Q2. Generate response in the format: The answer to Q1 is [answer], the answer to Q2 is [answer]."
    },
    'chartqa_prompt_fullstop': {
        "text_prompt": " 'Text': 'Q1': [[q1]], 'Q2': [[q2]]",
        "instruction_prompt": " First solve Q1 and then solve Q2. Generate response in the format: The answer to Q1 is [answer]. The answer to Q2 is [answer]."
    }
}