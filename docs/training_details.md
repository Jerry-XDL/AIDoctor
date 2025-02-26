# Training Detail

### Stage 1: Continue Pretraining

Phase One: PT (Continue PreTraining) Incremental Pre-training

Utilize datasets composed of encyclopedic and documentary materials to conduct incremental or secondary pre-training on domain-specific datasets. The objective is to infuse the model with domain-specific knowledge. Taking the medical field as an example, the aim of incremental pre-training is to enable the model to comprehend the symptoms, etiology, therapeutic drugs, treatment methods, and drug efficacy of conditions such as the common cold. This foundational knowledge is intended to facilitate the activation of such intrinsic knowledge during the subsequent SFT (Supervised Fine-Tuning) phase.

It is important to note that large models like GPT-3 and LLaMA can theoretically benefit from incremental pre-training. However, this process necessitates two critical conditions: 1) high-quality pre-training samples, and 2) substantial computational resources, with high demands on video memory. Even with the application of LoRA technology, it is essential to accommodate text of block_size=1024 or 2048 length into the video memory.

Furthermore, if the data utilized in your project has already been employed in the model's pre-training phase—such as Wikipedia or ArXiv, which were used in the pre-training of the LLaMA model—it is redundant to feed this data to LLaMA for incremental pre-training. Moreover, if the quality of the pre-training samples is not sufficiently high, it may potentially impair the generative capabilities of the original model.

Tips: The PT phase is optional and should be approached with caution.

Based on the llama-7b model, continue pre-training with medical encyclopedic data, with the expectation of injecting medical knowledge into the pre-trained model, resulting in the llama-7b-pt model.

Continue pretraining of the base llama-7b model to create llama-7b-pt:

```shell
cd scripts
sh run_pt.sh
```

[Training Parameters Explanation Wiki]

-   If your video memory is insufficient, you can reduce the batch_size to 1 and set block_size to 512 (this affects the maximum context length for training);

-   If you have more video memory, you can increase the block_size to 2048, which is the original pre-training length for llama and cannot be larger; also consider increasing the batch_size.

### Stage 2: Supervised FineTuning

Phase Two: SFT (Supervised Fine-tuning) Supervised Fine-tuning

Based on the llama-7b-pt model, supervised fine-tuning is performed using medical Q&A data to obtain the llama-7b-sft model.

Supervised fine-tuning of the base llama-7b-pt model to create llama-7b-sft.

```shell
cd scripts
sh run_sft.sh
```

### Stage 3: Reward Modeling

Phase Three: RM (Reward Model) Reward Model Construction

The RM (Reward Model), in principle, allows us to directly use human annotations to fine-tune the model with RLHF (Reinforcement Learning from Human Feedback).

However, this approach would require sending some samples to humans for scoring after each round of optimization. This process is costly and slow due to the large number of training samples needed for convergence and the limited speed at which humans can read and annotate. A strategy superior to direct feedback is to train a reward model RM with a set of human annotations before entering the RL loop. The purpose of the reward model is to simulate human scoring of text.

The best practice for constructing a reward model is to predict the ranking of outcomes, that is, for each prompt (input text) corresponding to two outcomes (yk, yj), the model predicts which one would receive a higher score from human annotators. The RM model is trained by manually scoring the outcomes of the SFT model, aiming to replace manual scoring. Essentially, it is a regression model designed to align with human preferences, primarily adhering to the "HHH" principle, specifically "helpful, honest, harmless."

Based on the llama-7b-sft model, use medical Q&A preference data to train a reward preference model, resulting in the llama-7b-reward model.

Reward modeling using dialog pairs from the reward dataset using the llama-7b-sft to create llama-7b-reward:

```shell
cd scripts
sh run_rm.sh
```

### Stage 4: Reinforcement Learning

Phase Four: RL (Reinforcement Learning) Reinforcement Learning with Human Feedback (RLHF)

The goal of the RL (Reinforcement Learning) model is to maximize the output of the reward model. Based on the steps above, we now have a fine-tuned language model (llama-7b-sft) and a reward model (llama-7b-reward), which allows us to commence the RL loop.

This process can be broadly divided into three steps:

Input a prompt, and the model generates a response.

Use the reward model to score the response.

Based on the score, perform a round of policy optimization reinforcement learning (PPO).

Conduct RL fine-tuning on the llama-7b-sft model using the llama-7b-reward model to obtain the llama-7b-rl model.

Reinforcement Learning fine-tuning of llama-7b-sft with the llama-7b-reward reward model to create llama-7b-rl

```shell
pip install git+https://github.com/lvwerra/trl
cd scripts
sh run_rl.sh
```
