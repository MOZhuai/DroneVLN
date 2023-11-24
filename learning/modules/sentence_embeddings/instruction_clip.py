from learning.modules.cuda_module import CudaModule as ModuleBase
from transformers import CLIPProcessor, CLIPModel


class InstructionCLIP(ModuleBase):
    def __init__(self):
        super(InstructionCLIP).__init__()
        self.last_output = None
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def init_weights(self):
        pass

    def reset(self):
        self.last_output = None

    def get(self):
        return self.last_output

    def forward(self, word_ids):
        inputs = self.processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt",
                           padding=True)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        batch_size = len(word_ids)
        sentence_embeddings = self.sentence_trans.encode(word_ids, convert_to_tensor=True)
        self.last_output = sentence_embeddings[batch_size - 1:batch_size]
        return sentence_embeddings

