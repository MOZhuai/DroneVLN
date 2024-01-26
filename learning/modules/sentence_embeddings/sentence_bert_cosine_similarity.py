import torch
from torch.autograd import Variable
from data_io.instructions import debug_untokenize_instruction
from sentence_transformers import SentenceTransformer
from learning.inputs.sequence import sequence_list_to_tensor
from learning.modules.cuda_module import CudaModule as ModuleBase

# TODO: Parametrize
VOCAB_SIZE = 2080


class SentenceBert(ModuleBase):

    def __init__(self):
        super(SentenceBert, self).__init__()
        self.last_output = None
        self.sentence_trans = SentenceTransformer('all-MiniLM-L12-v2')

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

    def forward(self, instructions):
        batch_size = len(instructions)
        sent_emb = self.sentence_trans.encode(instructions, convert_to_tensor=True)
        instructions_with_step = self.add_step_into_text(instructions)
        sent_emb_with_step = self.sentence_trans.encode(instructions_with_step, convert_to_tensor=True)
        self.last_output = sent_emb[batch_size-1:batch_size]
        return sent_emb, sent_emb_with_step
