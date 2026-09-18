def get_filename_for_experiment(config, **kwargs):
    """
    How do we want to save the files
    [Config]
        - experiment_name
        - model
        - prompt_template
    [File Being Loaded]
        - Fruit-Sport
            - Fruit category
            - Sport Category
            - sample_id
        - Fruit-Math / Math-Fruit
            - Fruit category
            - Math Answer
            - sample_id
    """
    experiment_name = config['experiment_name']
    model_name = config['model']['model_id'].replace("/", "-")
    prompt_template = config['task']['prompt_template']
    image_signal = config['task'].get('image_type', "actual")

    if experiment_name == "fruit_sport":
        fruit_category = kwargs.get('fruit_category', 'unknown_fruit')
        sport_category = kwargs.get('sport_category', 'unknown_sport')
        sample_id = kwargs.get('sample_id', 'unknown_id')
        filename = f"{experiment_name}__image-{image_signal}__{prompt_template}__fruit-{fruit_category}__sport-{sport_category}__id-{sample_id}"
    elif experiment_name in ["fruit_math", "math_fruit", "fruit_math_boosting"]:
        fruit_category = kwargs.get('fruit_category', 'unknown_fruit')
        math_answer = kwargs.get('math_answer', 'unknown_answer')
        sample_id = kwargs.get('sample_id', 'unknown_id')
        filename = f"{experiment_name}__image-{image_signal}__{prompt_template}__fruit-{fruit_category}__math-{math_answer}__id-{sample_id}"
    elif experiment_name == "vsr":
        sample_id = kwargs.get('sample_id', 'unknown_id')
        label = kwargs.get('label', None)
        suffix = ""
        if label is not None:
            suffix = f"_{label}"
        filename = f"{experiment_name}__image-{image_signal}__{prompt_template}__id-{sample_id}{suffix}"
    elif experiment_name == "chartqa_analysis":
        sample_id = kwargs.get('sample_id', 'unknown_id')
        filename = f"{experiment_name}__image-{image_signal}__{prompt_template}__id-{sample_id}"
    else:
        filename = f"{experiment_name}__image-{image_signal}__{prompt_template}__unknown_experiment"

    return filename


def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)