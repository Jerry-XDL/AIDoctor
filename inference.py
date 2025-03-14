# -*- coding: utf-8 -*-
""" 
@description: 
"""
import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
)

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--data_file', default=None, type=str,
                        help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--with_prompt', action='store_true', help="wrap the input with the prompt automatically")
    parser.add_argument('--interactive', action='store_true', help="run in the instruction mode (single-turn)")
    parser.add_argument('--predictions_file', default='./predictions.json', type=str)
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    args = parser.parse_args()
    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    generation_config = dict(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.3,
        max_new_tokens=400
    )

    # The prompt template below is taken from llama.cpp
    # and is slightly different from the one used in training.
    # But we find it gives better results
    prompt_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
    )

    sample_data = ["乙肝和丙肝的区别？"]

    def generate_prompt(instruction, input=None):
        if input:
            instruction = instruction + '\n' + input
        return prompt_input.format_map({'instruction': instruction})

    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None and os.path.exists(
            os.path.join(args.lora_model, "tokenizer_config.json")):
        args.tokenizer_path = args.lora_model
    else:
        args.tokenizer_path = args.base_model

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    base_model = model_class.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True,
    )

    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model

    if device == torch.device('cpu'):
        model.float()
    # test data
    if args.data_file is None:
        examples = sample_data
    else:
        with open(args.data_file, 'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)
    model.eval()

    with torch.no_grad():
        if args.interactive:
            print("Start inference with instruction mode.")

          
            print('-' * 85)
            print("+ This mode only supports single-turn QA.\n"
                  "+ If you want to experience multi-turn dialogue, please use llama.cpp or llamachat.")
            print('=' * 85)

            while True:
                raw_input_text = input("Input:")
                if len(raw_input_text.strip()) == 0:
                    break
                if args.with_prompt:
                    input_text = generate_prompt(instruction=raw_input_text)
                else:
                    input_text = raw_input_text
                inputs = tokenizer(input_text, return_tensors="pt")
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                output = tokenizer.decode(s, skip_special_tokens=True)
                if args.with_prompt:
                    response = output.split("### Response:")[1].strip()
                else:
                    response = output
                print("Response: ", response)
                print("\n")
        else:
            print("Start inference.")
            results = []
            for index, example in enumerate(examples):
                if args.with_prompt is True:
                    input_text = generate_prompt(instruction=example)
                else:
                    input_text = example
                inputs = tokenizer(input_text, return_tensors="pt")
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                output = tokenizer.decode(s, skip_special_tokens=True)
                if args.with_prompt:
                    response = output.split("### Response:")[1].strip()
                else:
                    response = output
                print(f"======={index}=======")
                print(f"Input: {example}\n")
                print(f"Output: {response}\n")

                results.append({"Input": input_text, "Output": response})

            dirname = os.path.dirname(args.predictions_file)
            os.makedirs(dirname, exist_ok=True)
            with open(args.predictions_file, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            with open(dirname + '/generation_config.json', 'w') as f:
                json.dump(generation_config, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
