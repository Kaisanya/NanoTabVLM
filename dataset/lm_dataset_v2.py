from PIL import Image
import torch
from model.model_nanotabvlm_v2 import NanoTabVLMV2
import os
from .lm_dataset import TabVLMDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TabVLMDatasetV2(TabVLMDataset):

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_paths = sample['image']
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        image_tensors = []
        for image_name in image_paths.split(','):
            image_name = image_name.strip()
            image = Image.open(f'{self.images_path}/{image_name}')
            image_tensor = NanoTabVLMV2.image2tensor(image, self.preprocess)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors, dim=0)

        return X, Y, loss_mask, image_tensors
