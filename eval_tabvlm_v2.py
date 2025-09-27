import argparse
import os
import random
import numpy as np
import torch
import warnings
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_nanotabvlm_v2 import NanoTabVLMV2, NanoTabVLMConfig

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(vlm_config, device, model_path):
    tokenizer = AutoTokenizer.from_pretrained('./model', use_fast=True)
    # 加载纯语言模型权重
    kwargs = {
        "input_size": 1024,
        "train_interpolation": "bicubic",
        "training": False,
    }
    model = NanoTabVLMV2(vlm_config, **kwargs)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    print(f'VLM参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    vision_model, preprocess = model.vision_encoder, model.processor
    return model.eval().to(device), tokenizer, vision_model.eval().to(device), preprocess


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table recognition of NanoTabVLM")
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.65, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--hidden_size', default=384, type=int)
    parser.add_argument('--num_hidden_layers', default=6, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--vocab_size', default=151936, type=int)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--model_path', default="checkpoint/nano_v2_vlm_384.pth", type=str)
    args = parser.parse_args()

    vlm_config = NanoTabVLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        use_moe=False,
        image_special_token='@@@@' * 1024,
        image_ids=[62182] * 1024,
    )
    model, tokenizer, vision_model, preprocess = init_model(vlm_config, args.device, args.model_path)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def chat_with_vlm(prompt, pixel_values, image_names):
        messages = [{"role": "user", "content": prompt}]
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:]

        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(args.device)

        print(f'[Image]: {image_names}')
        print('🤖️: ', end='')
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=args.max_seq_len,
            num_return_sequences=1,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=args.top_p,
            temperature=args.temperature,
            pixel_values=pixel_values
        )

        response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        print('\n\n')


    image_dir = './images/eval'
    prompt = f"{model.params.image_special_token}"

    for image_file in os.listdir(image_dir):
        image = Image.open(os.path.join(image_dir, image_file)).convert('RGB')
        pixel_tensors = NanoTabVLMV2.image2tensor(image, preprocess).to(args.device).unsqueeze(0).unsqueeze(0)
        chat_with_vlm(prompt, pixel_tensors, image_file)
