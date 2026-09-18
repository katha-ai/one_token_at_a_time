import os
import json
import hashlib
import math
import random


def _load_tagged_tokens(base_dir, model_name, task_name, sample_name):
    file_path = os.path.join(
        base_dir,
        model_name,
        task_name,
        f"{sample_name}.json"
    )
    file_path = file_path.replace("_boosting", "")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Block indices file not found: {file_path}")

    with open(file_path, "r") as f:
        return json.load(f)


def get_num_generated_steps(base_dir, model_name, task_name, sample_name):
    """
    Returns the number of generation steps actually recorded for this sample's
    prior (vanilla) run, i.e. how many generation-step indices are valid to
    target for this specific sample. Used by `style.type: random` boosting/
    blocking so a randomly chosen step can't land past the point where this
    sample's generation actually stopped.
    """
    tagged_tokens = _load_tagged_tokens(base_dir, model_name, task_name, sample_name)
    return len(tagged_tokens)


def fetch_block_indices(
    base_dir,
    model_name,
    task_name,
    sample_name,
    config,
    return_mapping=False
):
    """
    Returns a list of generation step indices to block/boost based on POS tags.

    Args:
        base_dir (str): Base directory containing tagged outputs
        model_name (str)
        task_name (str)
        sample_name (str)
        config (dict): blocking/boosting config with keys:
            - pos_tag_to_block (str or list)
            - only_last (bool)
            - only_first (bool)
            - extra_steps_to_block (int): Number of additional steps to block after the matching POS tag.
        return_mapping (bool): If True, returns a list of tuples (step, tag).
                               Otherwise, returns a list of steps.

    Returns:
        List[int] or List[Tuple[int, str]]: generation step indices or step-tag mapping
    """
    tagged_tokens = _load_tagged_tokens(base_dir, model_name, task_name, sample_name)

    pos_tags_input = config.get("pos_tag_to_block")
    only_last = config.get("only_last", False)
    only_first = config.get("only_first", False)

    if only_last and only_first:
        raise ValueError("Cannot set both `only_last` and `only_first` to True")

    if pos_tags_input is None:
        raise ValueError("`pos_tag_to_block` must be specified in config")

    if isinstance(pos_tags_input, str):
        pos_tags_list = [pos_tags_input]
    else:
        pos_tags_list = pos_tags_input
    
    pos_tags_set = set(pos_tags_list)

    # collect all indices where tag matches
    matches = []
    for i, tok in enumerate(tagged_tokens):
        tag = tok.get("tag")
        if tag in pos_tags_set:
            matches.append((i+1, tag))

    if not matches:
        return []

    if only_last:
        # Get last occurrence of each tag
        last_of_each = {}
        for idx, tag in matches:
            last_of_each[tag] = (idx, tag)
        matches = sorted(last_of_each.values(), key=lambda x: x[0])
    elif only_first:
        # Get first occurrence of each tag
        first_of_each = {}
        for idx, tag in matches:
            if tag not in first_of_each:
                first_of_each[tag] = (idx, tag)
        matches = sorted(first_of_each.values(), key=lambda x: x[0])

    extra_steps = config.get("extra_steps_to_block", 0)
    final_matches = []
    if extra_steps > 0:
        for idx, tag in matches:
            for i in range(extra_steps + 1):
                final_matches.append((idx + i, tag))
    else:
        final_matches = matches

    if return_mapping:
        return final_matches
    else:
        # Return unique steps, sorted
        return sorted(list(set([m[0] for m in final_matches])))


def fetch_random_step_indices(
    base_dir,
    model_name,
    task_name,
    sample_name,
    config
):
    """
    Returns a reproducible, per-sample list of randomly chosen generation step
    indices - a "blind" control for `style.type: precomputed` targeted
    blocking/boosting, so an observed effect can be attributed to *targeting*
    a specific step rather than to boosting/blocking at all.

    Steps are drawn without replacement from [1, num_generated_steps], where
    num_generated_steps is this sample's actual recorded generation length
    (from the same tagged-output JSON precomputed styles read) - so a step
    that never occurred for this sample can never be selected.

    When `chunk_to_boost` is a list (e.g. ['image', 'text']), the caller
    zips the returned steps positionally to the chunks. `sort_random`
    controls what order they come back in for that zip:
      - True (default): steps are sorted ascending, so the earlier step
        always maps to the first-listed chunk (e.g. image always boosted
        before text) - mirrors a targeted run's fixed modality ordering
        while still randomizing *when* each boost fires.
      - False: steps are left in the order `random.sample` drew them
        (itself randomized, not sorted) - so which chunk gets the earlier
        vs. later step is also random per sample, not just the step values.

    Args:
        config (dict): boosting/blocking `style` config with keys:
            - random_seed (int, required): base seed; combined with
              `sample_name` so each sample gets an independent but
              reproducible draw (not the same step(s) for every sample).
            - num_steps (int, default 1): how many distinct steps to draw.
            - sort_random (bool, default True): sort the drawn steps
              ascending before returning (see above).

    Returns:
        List[int]: unique generation step indices, sorted ascending if
        `sort_random` is True (default), otherwise in draw order.
    """
    num_generated_steps = get_num_generated_steps(base_dir, model_name, task_name, sample_name)
    if num_generated_steps < 1:
        return []

    random_seed = config.get("random_seed")
    if random_seed is None:
        raise ValueError("`random_seed` must be specified in config for reproducibility")

    num_steps = min(config.get("num_steps", 1), num_generated_steps)
    sort_random = config.get("sort_random", True)

    # Per-sample seed: the same base seed always yields the same draw for a
    # given sample, but samples don't all land on the same step(s).
    sample_seed = int(hashlib.md5(f"{random_seed}_{sample_name}".encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(sample_seed)

    drawn_steps = rng.sample(range(1, num_generated_steps + 1), num_steps)
    return sorted(drawn_steps) if sort_random else drawn_steps


def fetch_partial_random_step_indices(
    base_dir,
    model_name,
    task_name,
    sample_name,
    config
):
    """
    A "controlled random" variant of `fetch_random_step_indices`: instead of
    drawing from the whole generation range, restricts the draw to the first
    `fraction` of the steps matching a given POS tag - e.g. "boost somewhere
    in the first half of FRUIT_INTRO" rather than "boost anywhere in the
    whole response". Used for negative-control boosting arms that need to
    land in a specific pre-answer region without ever risking landing on the
    answer token itself (FRUIT_CONCEPT / MATH_ANSWER), while still varying
    *which* step within that region gets boosted per sample.

    When `pos_tag_to_block` is a list (mirrors `style.type: precomputed`'s
    list handling - e.g. boosting image during FRUIT_INTRO and text during
    MATH_INTRO in the same run), one step is drawn independently per tag,
    each with its own per-sample-per-tag seed so the two draws aren't
    correlated.

    Args:
        config (dict): boosting/blocking `style` config with keys:
            - pos_tag_to_block (str or list[str], required): tag(s) whose
              matched-step span to restrict the draw to.
            - fraction (float or dict[str, float], default 0.5): take the
              first ceil(fraction * len(matches)) matched steps for a tag (at
              least 1), then draw one uniformly from that prefix. A dict
              (tag -> fraction) sets a per-tag margin - e.g. a tighter
              fraction for a tag immediately followed by a sensitive region
              (MATH_INTRO -> MATH_ANSWER) than for one with no such neighbor
              (FRUIT_INTRO, which is always first in this prompt template so
              has no preceding boost that could drift its alignment). A tag
              missing from the dict falls back to 0.5.
            - random_seed (int, required): base seed; combined with
              `sample_name` and the tag so different tags/samples don't draw
              in a correlated way.

    Returns:
        dict: {step: tag} - one entry per tag in `pos_tag_to_block` that has
        at least one match for this sample (a tag with zero matches is
        silently skipped, same as `fetch_block_indices` returning `[]`).
    """
    tagged_tokens = _load_tagged_tokens(base_dir, model_name, task_name, sample_name)

    pos_tags_input = config.get("pos_tag_to_block")
    if pos_tags_input is None:
        raise ValueError("`pos_tag_to_block` must be specified in config")
    pos_tags_list = [pos_tags_input] if isinstance(pos_tags_input, str) else pos_tags_input

    random_seed = config.get("random_seed")
    if random_seed is None:
        raise ValueError("`random_seed` must be specified in config for reproducibility")

    fraction_input = config.get("fraction", 0.5)

    step_to_tag = {}
    for tag in pos_tags_list:
        fraction = fraction_input.get(tag, 0.5) if isinstance(fraction_input, dict) else fraction_input
        if not (0 < fraction <= 1):
            raise ValueError(f"`fraction` for tag {tag!r} must be in (0, 1], got {fraction}")

        matches = [i + 1 for i, tok in enumerate(tagged_tokens) if tok.get("tag") == tag]
        if not matches:
            continue

        prefix_len = max(1, math.ceil(len(matches) * fraction))
        prefix = matches[:prefix_len]

        sample_seed = int(hashlib.md5(f"{random_seed}_{sample_name}_{tag}".encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(sample_seed)
        step_to_tag[rng.choice(prefix)] = tag

    return step_to_tag
