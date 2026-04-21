import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description='Smoke test for local Qwen2.5-0.5B model')
    parser.add_argument('--model-path', type=str, default='/root/adaptive env selection/model/Qwen2.5-0.5B')
    parser.add_argument('--prompt', type=str, default='请用两句话解释什么是强化学习。')
    parser.add_argument('--max-new-tokens', type=int, default=120)
    args = parser.parse_args()

    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32

    print(f'[INFO] torch={torch.__version__}, cuda={has_cuda}, dtype={dtype}')
    print(f'[INFO] loading model from: {args.model_path}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map='auto' if has_cuda else None,
        trust_remote_code=True,
    )

    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors='pt')
    else:
        inputs = tokenizer(args.prompt, return_tensors='pt')

    if has_cuda:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )

    new_ids = output_ids[0][inputs['input_ids'].shape[-1]:]
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    print('\n[OUTPUT]')
    print(output_text.strip())


if __name__ == '__main__':
    main()
