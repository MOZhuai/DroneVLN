import torch
from torch.autograd import Variable
from data_io.instructions import debug_untokenize_instruction
from sentence_transformers import SentenceTransformer
from learning.inputs.sequence import sequence_list_to_tensor
from learning.modules.cuda_module import CudaModule as ModuleBase

# TODO: Parametrize
VOCAB_SIZE = 2080


class SentenceBert(ModuleBase):

    def __init__(self, word_embedding_size, embed_size, lstm_layers=1, run_name=""):
        super(SentenceBert, self).__init__()
        self.last_output = None
        self.sentence_trans = SentenceTransformer('all-MiniLM-L12-v2')
        # self.sentence_trans = SentenceTransformer('paraphrase-albert-small-v2')

    def init_weights(self):
        pass

    def reset(self):
        self.last_output = None

    def get(self):
        return self.last_output

    def forward(self, word_ids):
        batch_size = len(word_ids)
        sentence_embeddings = self.sentence_trans.encode(word_ids, convert_to_tensor=True)
        self.last_output = sentence_embeddings[batch_size-1:batch_size]
        return sentence_embeddings
