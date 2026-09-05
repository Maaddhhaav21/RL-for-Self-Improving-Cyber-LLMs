from collections import Counter

import torch


def repetition_penalty(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Calculate a penalty based on repeated tokens.

    Higher values mean the generated sequence contains
    more excessive token repetition.

    Args:
        token_ids:
            Tensor of shape:
            [batch_size, sequence_length]

        attention_mask:
            Optional tensor with the same shape.
            1 = valid token
            0 = padding token

    Returns:
        Tensor containing one repetition penalty per sequence.
    """

    batch_size = token_ids.size(0)
    penalties = []

    for batch_index in range(batch_size):
        tokens = token_ids[batch_index]

        if attention_mask is not None:
            mask = attention_mask[batch_index].bool()
            tokens = tokens[mask]

        if len(tokens) == 0:
            penalties.append(0.0)
            continue

        unique_tokens = torch.unique(tokens)

        total_tokens = len(tokens)
        unique_count = len(unique_tokens)

        repeated_tokens = total_tokens - unique_count

        penalty = repeated_tokens / max(total_tokens, 1)

        penalties.append(penalty)

    return torch.tensor(
        penalties,
        dtype=torch.float32,
        device=token_ids.device,
    )


def consecutive_repetition_penalty(
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize consecutive repeated tokens.

    Example:
        hello hello hello

    This type of repetition can indicate poor-quality
    or degenerate generation.
    """

    batch_size = token_ids.size(0)
    penalties = []

    for batch_index in range(batch_size):
        tokens = token_ids[batch_index]

        if len(tokens) <= 1:
            penalties.append(0.0)
            continue

        repeated_count = 0

        for index in range(1, len(tokens)):
            if tokens[index] == tokens[index - 1]:
                repeated_count += 1

        penalty = repeated_count / max(len(tokens) - 1, 1)

        penalties.append(penalty)

    return torch.tensor(
        penalties,
        dtype=torch.float32,
        device=token_ids.device,
    )


def repeated_ngram_penalty(
    token_ids: torch.Tensor,
    ngram_size: int = 3,
) -> torch.Tensor:
    """
    Penalize repeated n-grams.

    For example, if the same sequence of 3 tokens appears
    multiple times, the penalty increases.
    """

    batch_size = token_ids.size(0)
    penalties = []

    for batch_index in range(batch_size):
        tokens = token_ids[batch_index].tolist()

        if len(tokens) < ngram_size:
            penalties.append(0.0)
            continue

        ngrams = []

        for index in range(
            len(tokens) - ngram_size + 1
        ):
            ngram = tuple(
                tokens[index:index + ngram_size]
            )

            ngrams.append(ngram)

        counts = Counter(ngrams)

        repeated_count = sum(
            count - 1
            for count in counts.values()
            if count > 1
        )

        penalty = repeated_count / max(len(ngrams), 1)

        penalties.append(penalty)

    return torch.tensor(
        penalties,
        dtype=torch.float32,
        device=token_ids.device,
    )


def length_penalty(
    token_ids: torch.Tensor,
    min_length: int = 10,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Penalize generations that are extremely short or
    unnecessarily long.
    """

    lengths = torch.full(
        (token_ids.size(0),),
        token_ids.size(1),
        dtype=torch.float32,
        device=token_ids.device,
    )

    penalties = torch.zeros_like(lengths)

    too_short = lengths < min_length
    too_long = lengths > max_length

    penalties[too_short] = (
        (min_length - lengths[too_short])
        / min_length
    )

    penalties[too_long] = (
        (lengths[too_long] - max_length)
        / max_length
    )

    return penalties


def compute_generation_penalty(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    repetition_weight: float = 1.0,
    consecutive_weight: float = 1.0,
    ngram_weight: float = 1.0,
) -> torch.Tensor:
    """
    Combine all generation-quality penalties.

    This final value can be subtracted from the reward
    during attacker policy optimization.

    Example:

        final_reward = task_reward - penalty
    """

    token_penalty = repetition_penalty(
        token_ids=token_ids,
        attention_mask=attention_mask,
    )

    consecutive_penalty = (
        consecutive_repetition_penalty(
            token_ids=token_ids
        )
    )

    ngram_penalty = repeated_ngram_penalty(
        token_ids=token_ids,
        ngram_size=3,
    )

    total_penalty = (
        repetition_weight * token_penalty
        + consecutive_weight * consecutive_penalty
        + ngram_weight * ngram_penalty
    )

    return total_penalty