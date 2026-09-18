import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import yaml
import pandas as pd
from tqdm import tqdm
import logging
from utils.logging_setup import setup_logging

from models.llava_onevision import LlavaOnevisionHandler
from models.qwen2_llm import Qwen2LLMHandler
from models.qwen_25_vl import QwenVLHandler
from models.qwen_35_vl import Qwen35VLHandler
from models.gemma_3 import GemmaHandler
from models.gemma_4_omni import Gemma4OmniHandler
from models.gemma_4 import Gemma4Handler
from models.qwen_25_omni import Qwen25OmniHandler
from tasks.fruit_sports import FruitSportsTask
from tasks.fruit_math import FruitMathTask
from tasks.vsr import VSRTask
from tasks.chartqa import ChartQATask

from analysis.extractor import AttentionExtractor
# from analysis.extractor_aggregated import AttentionExtractor
import sys
from prompts.task_prompt_templates import prompt
import argparse
from utils.generic_utils import get_filename_for_experiment
from utils.fetch_block_indices import fetch_block_indices, fetch_random_step_indices, fetch_partial_random_step_indices
# from utils.fetch_block_indices_faulty import fetch_block_indices_faulty as fetch_block_indices
os.environ["HF_OFFLINE"] = "1"

def main(config_path="configs/fruit_sport.yaml"):
    # 1. Load config and setup logging
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    setup_logging(config['save_dir'])
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {config['experiment_name']}")

    # 2. Load Data
    df = pd.read_csv(config['data']['csv_path'])
    if 'image_path' not in df.columns and 'image_name' in df.columns:
        df['image_path'] = df['image_name']

    
    # 3. Setup Model Handler
    model_type = config['model']['type']
    if model_type == "llava_onevision":
        model_handler = LlavaOnevisionHandler(config['model'])
    elif model_type == "qwen2_llm":
        model_handler = Qwen2LLMHandler(config['model'])
    elif model_type == "qwen_25_vl":
        model_handler = QwenVLHandler(config['model'])
    elif model_type == "qwen_35_vl":
        model_handler = Qwen35VLHandler(config['model'])
    elif model_type == "gemma_3":
        model_handler = GemmaHandler(config['model'])
    elif model_type == "gemma_4_omni":
        model_handler = Gemma4OmniHandler(config['model'])
    elif model_type == "gemma_4":
        model_handler = Gemma4Handler(config['model'])
    elif model_type == "qwen_25_omni":
        model_handler = Qwen25OmniHandler(config['model'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_handler.load_model()
    logger.info(f"Loaded model: {config['model']['model_id']}")

    # 4. Setup Task
    task_name = config['task']['name']
    if task_name == "fruit_sport":
        task = FruitSportsTask(config['task'], prompt_templates=prompt)

    elif task_name == "fruit_math":
        task = FruitMathTask(config['task'], prompt_templates=prompt)

    elif task_name == "vsr":
        task = VSRTask(config['task'], prompt_templates=prompt)

    elif task_name == "chartqa":
        task = ChartQATask(config['task'], prompt_templates=prompt)

    else:
        raise ValueError(f"Unknown task: {task_name}")

    # 5. Setup the Extractor
    extractor = AttentionExtractor(model_handler, task, config)

    # 6. Run the loop
    logger.info(f"Starting attention extraction for {len(df)} samples.")
    for sample_idx, item in tqdm(df.iterrows(), total=len(df), desc="Extracting Attentions"):
        try:
            
            base_name = get_filename_for_experiment(config=config,
                                            fruit_category = item.get('fruit_category', 'unknown_fruit'),
                                            sport_category = item.get('sport_category', 'unknown_sport'),
                                            sample_id = sample_idx,
                                            math_answer = item.get('math_answer', 'unknown_answer'),
                                            query_item = item.get('query_item', 'unknown_query'),
                                            image_count = item.get('image_count', 'unknown_image_count'),
                                            text_count = item.get('text_count', 'unknown_text_count'),
                                            label = item.get('label'),
                                            mma_original_id = item.get('original_id'),
                                            mma_answer = item.get('answer'),
                                            mms_class_label = item.get('class_label')
                                            )
            
            # if base_name != "vsr__image-actual__vsr_prompt_discrepancy_updated__id-44_1":
            #     continue
            
            import os
            net_file_path = os.path.join(config['save_dir'], "0", "attention_progression",base_name + ".npz")
            if os.path.exists(net_file_path):
                logger.info(f"Attention data for sample {sample_idx} already exists at {net_file_path}. Skipping extraction.")
                continue
            
            if config.get('blocking') and config['blocking'].get('use_blocking'):
                if config['blocking']['style']['type'] == 'precomputed':
                    block_list = fetch_block_indices(base_dir=config['blocking']['style']['pos_dir'],
                                                        model_name=config['blocking']['style']['model'],
                                                        task_name=task_name,
                                                        sample_name=base_name,
                                                        config = config['blocking']['style'])
                    logger.info(f"Using Precomputed blocking config for sample {sample_idx}: {block_list}")
                    debug_blocking_config = {
                        'block_type': 'layer_collapsed',  # Apply to all layers
                        'step_to_block': block_list, 
                        'chunk': config['blocking']['chunk_to_block'], # The target to block (ensure this key exists in your chunk ranges)
                        'blocking_method': config['blocking']['method'] #lazy or total
                    }
                elif config['blocking']['style']['type'] == 'dynamic':
                    logger.info(f"Using Dynamic blocking config for sample {sample_idx}: stop_on_token={config['blocking']['style']['stop_on_token']}, chunk_to_block={config['blocking']['chunk_to_block']}, block_next_occurrence={config['blocking']['style'].get('block_next_occurrence', False)}")
                    debug_blocking_config = {
                        'block_type': 'layer_collapsed',  # Apply to all layers
                        'stop_on_token': config['blocking']['style']['stop_on_token'],               # Block during the generation of the 1st new token #1 implies the first step
                        'chunk': config['blocking']['chunk_to_block'],
                        'blocking_method': config['blocking']['method'], #lazy or total
                        'block_next_occurrence': config['blocking']['style'].get('block_next_occurrence', False)
                    }
                elif config['blocking']['style']['type'] == 'all_steps':
                    logger.info(f"Using All Steps blocking config for sample {sample_idx}: chunk_to_block={config['blocking']['chunk_to_block']}")
                    debug_blocking_config = {
                        'block_type': 'layer_collapsed',  # Apply to all layers
                        'step_to_block': list(range(2, 25)),
                        'blocking_method': config['blocking']['method'], #lazy or total
                        'chunk': config['blocking']['chunk_to_block']                 # The target to block (ensure this key exists in your chunk ranges)
                    }
                else:
                    raise ValueError(f"Unknown blocking style: {config['blocking']['style']['type']}")
            else:
                debug_blocking_config = None

            # --- BOOSTING CONFIG ---
            if config.get('boosting') and config['boosting'].get('use_boosting'):
                if config['boosting']['style']['type'] == 'precomputed':
                    boost_info = fetch_block_indices(base_dir=config['boosting']['style']['pos_dir'],
                                                        model_name=config['boosting']['style']['model'],
                                                        task_name=task_name,
                                                        sample_name=base_name,
                                                        config = config['boosting']['style'],
                                                        return_mapping=True)
                    
                    chunk_to_boost = config['boosting']['chunk_to_boost']
                    pos_tag_to_block = config['boosting']['style']['pos_tag_to_block']
                    
                    step_to_chunk = {}
                    if isinstance(chunk_to_boost, list):
                        if not isinstance(pos_tag_to_block, list) or len(pos_tag_to_block) != len(chunk_to_boost):
                            raise ValueError("chunk_to_boost and pos_tag_to_block must be lists of the same length when boosting multiple chunks")
                        
                        tag_to_chunk = {tag: chunk for tag, chunk in zip(pos_tag_to_block, chunk_to_boost)}
                        for step, tag in boost_info:
                            step_to_chunk[step] = tag_to_chunk.get(tag)
                    else:
                        for step, tag in boost_info:
                            step_to_chunk[step] = chunk_to_boost

                    logger.info(f"Using Precomputed boosting config for sample {sample_idx}: {step_to_chunk}")
                    debug_boosting_config = {
                        'boost_type': config['boosting'].get('type', 'logit'),
                        'boost_factor': config['boosting']['boost_factor'],
                        'layers_to_boost': config['boosting'].get('layers_to_boost', 'all'),
                        'step_to_boost': step_to_chunk,
                        'chunk': None if isinstance(chunk_to_boost, list) else chunk_to_boost
                    }
                elif config['boosting']['style']['type'] == 'precomputed_partial_random':
                    # "Controlled random" style: like `precomputed`, but each
                    # tag's step is drawn uniformly from the first
                    # `style.fraction` (default 50%) of that tag's matched
                    # steps, instead of always taking the first occurrence.
                    # E.g. chunk_to_boost: ['image', 'text'],
                    # pos_tag_to_block: ['FRUIT_INTRO', 'MATH_INTRO'] boosts
                    # image somewhere in the first half of FRUIT_INTRO and
                    # text somewhere in the first half of MATH_INTRO -
                    # varying *where* within a safely-non-answer region each
                    # boost fires, while never risking FRUIT_CONCEPT /
                    # MATH_ANSWER.
                    style_cfg = config['boosting']['style']
                    chunk_to_boost = config['boosting']['chunk_to_boost']
                    pos_tag_to_block = style_cfg['pos_tag_to_block']

                    step_to_tag = fetch_partial_random_step_indices(
                        base_dir=style_cfg['pos_dir'],
                        model_name=style_cfg['model'],
                        task_name=task_name,
                        sample_name=base_name,
                        config=style_cfg
                    )

                    step_to_chunk = {}
                    if isinstance(chunk_to_boost, list):
                        if not isinstance(pos_tag_to_block, list) or len(pos_tag_to_block) != len(chunk_to_boost):
                            raise ValueError("chunk_to_boost and pos_tag_to_block must be lists of the same length when boosting multiple chunks")

                        tag_to_chunk = {tag: chunk for tag, chunk in zip(pos_tag_to_block, chunk_to_boost)}
                        for step, tag in step_to_tag.items():
                            step_to_chunk[step] = tag_to_chunk.get(tag)
                    else:
                        for step, tag in step_to_tag.items():
                            step_to_chunk[step] = chunk_to_boost

                    logger.info(f"Using Precomputed-Partial-Random boosting config for sample {sample_idx}: step_to_chunk={step_to_chunk} (from step_to_tag={step_to_tag})")
                    debug_boosting_config = {
                        'boost_type': config['boosting'].get('type', 'logit'),
                        'boost_factor': config['boosting']['boost_factor'],
                        'layers_to_boost': config['boosting'].get('layers_to_boost', 'all'),
                        'step_to_boost': step_to_chunk,
                        'chunk': None if isinstance(chunk_to_boost, list) else chunk_to_boost
                    }
                elif config['boosting']['style']['type'] == 'dynamic':
                    logger.info(f"Using Dynamic boosting config for sample {sample_idx}: stop_on_token={config['boosting']['style']['stop_on_token']}, chunk_to_boost={config['boosting']['chunk_to_boost']}, boost_next_occurrence={config['boosting']['style'].get('boost_next_occurrence', False)}")
                    debug_boosting_config = {
                        'boost_type': config['boosting'].get('type', 'logit'),
                        'boost_factor': config['boosting']['boost_factor'],
                        'layers_to_boost': config['boosting'].get('layers_to_boost', 'all'),
                        'stop_on_token': config['boosting']['style']['stop_on_token'],
                        'chunk': config['boosting']['chunk_to_boost'],
                        'boost_next_occurrence': config['boosting']['style'].get('boost_next_occurrence', False)
                    }
                elif config['boosting']['style']['type'] == 'all_steps':
                    logger.info(f"Using All Steps boosting config for sample {sample_idx}: chunk_to_boost={config['boosting']['chunk_to_boost']}")
                    debug_boosting_config = {
                        'boost_type': config['boosting'].get('type', 'logit'),
                        'boost_factor': config['boosting']['boost_factor'],
                        'layers_to_boost': config['boosting'].get('layers_to_boost', 'all'),
                        'step_to_boost': list(range(1, 25)),
                        'chunk': config['boosting']['chunk_to_boost']
                    }
                elif config['boosting']['style']['type'] == 'random':
                    chunk_to_boost = config['boosting']['chunk_to_boost']
                    style_cfg = config['boosting']['style']

                    if isinstance(chunk_to_boost, list):
                        # Blind control for `precomputed` multi-chunk boosting (e.g.
                        # chunk_to_boost: ['image', 'text']): draw one random step per
                        # chunk instead of matching each chunk to its POS tag's step.
                        # Steps are drawn without replacement and sorted ascending, then
                        # zipped positionally to chunk_to_boost - so, like the targeted
                        # run where FRUIT_CONCEPT (image) precedes MATH_ANSWER (text),
                        # the earlier random step always gets the first-listed chunk.
                        # This preserves the targeted run's modality *ordering* while
                        # randomizing *when* each modality gets boosted, isolating
                        # whether targeting (not just boosting-somewhere) drives effects.
                        num_steps = style_cfg.get('num_steps', len(chunk_to_boost))
                        if num_steps != len(chunk_to_boost):
                            raise ValueError("`num_steps` must equal len(chunk_to_boost) when chunk_to_boost is a list for `random` boosting style")

                        random_steps = fetch_random_step_indices(base_dir=style_cfg['pos_dir'],
                                                            model_name=style_cfg['model'],
                                                            task_name=task_name,
                                                            sample_name=base_name,
                                                            config=style_cfg)
                        step_to_chunk = dict(zip(random_steps, chunk_to_boost))
                        logger.info(f"Using Random boosting config for sample {sample_idx}: step_to_chunk={step_to_chunk}")
                        debug_boosting_config = {
                            'boost_type': config['boosting'].get('type', 'logit'),
                            'boost_factor': config['boosting']['boost_factor'],
                            'layers_to_boost': config['boosting'].get('layers_to_boost', 'all'),
                            'step_to_boost': step_to_chunk,
                            'chunk': None
                        }
                    else:
                        random_steps = fetch_random_step_indices(base_dir=style_cfg['pos_dir'],
                                                            model_name=style_cfg['model'],
                                                            task_name=task_name,
                                                            sample_name=base_name,
                                                            config=style_cfg)
                        logger.info(f"Using Random boosting config for sample {sample_idx}: steps={random_steps}, chunk_to_boost={chunk_to_boost}")
                        debug_boosting_config = {
                            'boost_type': config['boosting'].get('type', 'logit'),
                            'boost_factor': config['boosting']['boost_factor'],
                            'layers_to_boost': config['boosting'].get('layers_to_boost', 'all'),
                            'step_to_boost': random_steps,
                            'chunk': chunk_to_boost
                        }
                else:
                    raise ValueError(f"Unknown boosting style: {config['boosting']['style']['type']}")
            else:
                debug_boosting_config = None

            extractor.process_sample(item, sample_idx, blocking_dict=debug_blocking_config, boosting_dict=debug_boosting_config)
        except Exception as e:
            logger.error(f"Failed to process sample {sample_idx}: {e}", exc_info=True)
            continue
            # exc_info=True will log the full error stacktrace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attention analysis.")
    parser.add_argument('--config', type=str, default="configs/fruit_sport.yaml", help='Path to the config file.')
    args = parser.parse_args()
    config_path = args.config
    main(config_path)