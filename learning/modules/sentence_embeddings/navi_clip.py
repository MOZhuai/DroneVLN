import torch
import numpy as np
from data_io.instructions import debug_untokenize_instruction
from transformers import CLIPProcessor, CLIPModel
from learning.inputs.sequence import sequence_list_to_tensor
from learning.modules.cuda_module import CudaModule as ModuleBase

# TODO: Parametrize
VOCAB_SIZE = 2080


class NaviCLIP(ModuleBase):

    def __init__(self, word_embedding_size, embed_size, lstm_layers=1, run_name=""):
        super(NaviCLIP, self).__init__()
        self.last_output = None
        model_name = "openai/clip-vit-base-patch16"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def init_weights(self):
        pass

    def reset(self):
        self.last_output = None

    def get(self):
        return self.last_output

    def add_step_into_text(self, text_list):
        for i, text in enumerate(text_list):
            if (i + 1) % 10 == 1:
                text_list[i] = f"The pictures of {i+1} st step of navigation instruction {text}"
            elif (i + 1) % 10 == 2:
                text_list[i] = f"The pictures of {i+1} nd step of navigation instruction {text}"
            else:
                text_list[i] = f"The pictures of {i+1} th step of navigation instruction {text}"
        return text_list

    def forward(self, words_list, observations):
        words_with_step = words_list.copy()
        words_with_step = self.add_step_into_text(words_with_step)

        inputs_with_step = self.processor(words_with_step, images=observations, return_tensors="pt", padding=True, truncation=True).to(device="cuda")
        output_with_step = self.model(**inputs_with_step)

        sentence_embeddings_with_step = output_with_step.text_embeds
        image_encoding = output_with_step.image_embeds

        # inputs = self.processor(words_list, images=observations, return_tensors="pt", padding=True, truncation=True).to(device="cuda")
        # output = self.model(**inputs)
        # sentence_embeddings = output.text_embeds

        return sentence_embeddings_with_step, sentence_embeddings_with_step, image_encoding
        # return sentence_embeddings, sentence_embeddings_with_step, image_encoding
