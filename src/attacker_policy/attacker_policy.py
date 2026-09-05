import torch
import torch.nn as nn

from .generation_penalties import (
    compute_generation_penalty,
)


class AttackerPolicy:
    """
    Policy-side implementation for the Attacker agent.

    The attacker model generates candidate prompt injections.

    This class performs the policy optimization component
    using a PPO-style clipped objective, which is the
    policy component used within the MAPPO framework.

    The centralized critic and advantage estimation are
    supplied by the shared multi-agent infrastructure.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_epsilon: float = 0.2,
        entropy_coefficient: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str | None = None,
    ):
        """
        Initialize the attacker policy.

        Args:
            model:
                The attacker language model.

            optimizer:
                Optimizer used to update the attacker model.

            clip_epsilon:
                PPO/MAPPO clipping coefficient.

            entropy_coefficient:
                Encourages exploration.

            max_grad_norm:
                Maximum gradient norm for gradient clipping.

            device:
                Device used for training.
        """

        self.model = model
        self.optimizer = optimizer

        self.clip_epsilon = clip_epsilon
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

    def compute_policy_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute the clipped PPO/MAPPO policy loss.

        Args:
            new_log_probs:
                Log probabilities from the current attacker policy.

            old_log_probs:
                Log probabilities from the previous attacker policy.

            advantages:
                Advantages calculated using the centralized critic.

            mask:
                Optional mask for valid tokens.

        Returns:
            policy_loss and metrics dictionary.
        """

        probability_ratio = torch.exp(
            new_log_probs - old_log_probs
        )

        unclipped_objective = (
            probability_ratio * advantages
        )

        clipped_ratio = torch.clamp(
            probability_ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        )

        clipped_objective = (
            clipped_ratio * advantages
        )

        objective = torch.minimum(
            unclipped_objective,
            clipped_objective,
        )

        if mask is not None:
            mask = mask.float()

            policy_loss = -(
                objective * mask
            ).sum() / mask.sum().clamp(min=1.0)

        else:
            policy_loss = -objective.mean()

        metrics = {
            "policy_loss": policy_loss.detach().item(),
            "ratio_mean": probability_ratio.mean()
            .detach()
            .item(),
        }

        return policy_loss, metrics

    def compute_entropy(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate policy entropy.

        Higher entropy encourages exploration and helps
        prevent the attacker policy from collapsing into
        identical generations.
        """

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        log_probabilities = torch.log_softmax(
            logits,
            dim=-1,
        )

        entropy = -torch.sum(
            probabilities * log_probabilities,
            dim=-1,
        )

        if mask is not None:
            mask = mask.float()

            return (
                entropy * mask
            ).sum() / mask.sum().clamp(min=1.0)

        return entropy.mean()

    def compute_adjusted_reward(
        self,
        base_reward: torch.Tensor,
        generated_token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply generation-quality penalties to the reward.

        The attacker receives a lower reward when its
        generated output contains excessive repetition
        or other degenerate generation patterns.
        """

        penalty = compute_generation_penalty(
            token_ids=generated_token_ids,
            attention_mask=attention_mask,
        )

        adjusted_reward = (
            base_reward - penalty
        )

        return adjusted_reward, penalty

    def update(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Perform one attacker policy update.

        The total loss consists of:

        1. PPO/MAPPO policy loss
        2. Entropy bonus for exploration
        """

        policy_loss, metrics = (
            self.compute_policy_loss(
                new_log_probs=new_log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages,
                mask=mask,
            )
        )

        entropy = self.compute_entropy(
            logits=logits,
            mask=mask,
        )

        total_loss = (
            policy_loss
            - self.entropy_coefficient * entropy
        )

        self.optimizer.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
        )

        self.optimizer.step()

        metrics.update(
            {
                "entropy": entropy.detach().item(),
                "total_loss": total_loss.detach().item(),
            }
        )

        return metrics