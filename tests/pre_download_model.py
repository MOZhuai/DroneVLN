from transformers import ViTImageProcessor, ViTForImageClassification

model_name = 'WinKawaks/vit-small-patch16-224'
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)