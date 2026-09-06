# Member 1 — Attacker Policy and Curation Plan

## 1. Introduction

This document describes the responsibilities, methodology, and implementation plan for the Attacker component in the **RL for Self-Improving Cyber LLMs** project.

Member 1 is responsible for preparing and maintaining the attacker-side data and policy foundation. The attacker is an important component of the adversarial learning system because it generates prompt-injection attempts that can be evaluated by the defender and other shared evaluation components.

The attacker component was initially developed through supervised fine-tuning during Review 1. For Review 2, the focus moves toward preparing the attacker policy for integration into the broader multi-agent reinforcement-learning architecture.

---

# 2. Member 1 Responsibilities

The primary responsibilities of Member 1 are:

1. Collect and curate attacker seed data.
2. Organize the attacker dataset.
3. Convert the raw attacker data into a format suitable for supervised fine-tuning.
4. Perform attacker model warmup using SFT.
5. Maintain the attacker-policy data pipeline.
6. Prepare the attacker component for integration with the MAPPO-based training system.
7. Consider generation quality so that the attacker does not produce meaningless or excessively repetitive outputs.
8. Document the attacker curation methodology and policy workflow.

---

# 3. Review 1 Work

During Review 1, the initial attacker training pipeline was implemented.

The overall pipeline was:

```text
Seed Injection Dataset
        ↓
Prompt Formatting
        ↓
formatted_attacks.jsonl
        ↓
Supervised Fine-Tuning
        ↓
Initial Attacker Model