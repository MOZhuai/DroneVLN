from transformers import ViTImageProcessor, ViTForImageClassification
from learning.modules.cuda_module import CudaModule as ModuleBase
from torch import nn


class Vit4Similarity(ModuleBase):
    def __init__(self):
        super().__init__()
        self.last_output = None
        model_name = 'WinKawaks/vit-small-patch16-224'
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)
        # self.model.classifier = nn.Linear(in_features=768, out_features=384, bias=True)
        self.model.classifier = nn.Identity()

    def init_weights(self):
        pass

    def reset(self):
        self.last_output = None

    def get(self):
        return self.last_output

    def forward(self, images):
        batch_size = len(images)
        inputs = self.feature_extractor(images=images, return_tensors="pt").to(device="cuda")
        outputs = self.model(**inputs)
        img_embs = outputs.logits

        for i in range(batch_size):
            if i == 0:
                continue
            img_embs[i] = img_embs[i] + img_embs[i-1]

        for i in range(batch_size):
            img_embs[i] = img_embs[i] / (i + 1)

        return img_embs
