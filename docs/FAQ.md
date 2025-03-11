Q: "NotImplementedError: Cannot copy out of meta tensor; no data!"
A: This error occurs due to insufficient memory on a single GPU. When device_map='auto' is used and the GPU memory is fully occupied, the system automatically attempts to load the model onto the CPU, leading to the _move_model_to_device error.
Solution: Specify multi-GPU training. For example, use CUDA_VISIBLE_DEVICES=0,1,2,3 python supervised_finetuning.py ... and adjust the batch size to be larger to fully utilize the GPU memory. This approach is similar to data parallelism and maximizes the use of GPUs to accelerate training. For more details.


Q: Errors when merging LoRA (peft) trained models of chatglm and baichuan
A: For chatglm and baichuan models, the code and weight files are stored together, and the code has not been integrated into the official transformers library. When merging LoRA, you need to copy all the Python files from the original weight path to the merged folder for use. For more information.

Q: Why can't chatglm and baichuan models be used for RM and RL training?
A: chatglm is not a standard CausalLM. The RM (Reward Modeling) phase requires AutoModelForSequenceClassification, which chatglm does not implement. PPO (Proximal Policy Optimization) training requires AutoModelForCausalLMWithValueHead, which chatglm also does not support. For the same reasons, the baichuan model cannot be used for RM and RL training either. These functionalities will be supported only after the official transformers library is compatible with chatglm and baichuan models. For further details.