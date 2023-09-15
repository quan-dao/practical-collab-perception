import torch
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation


feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")


for i in range(4):
    image = Image.open(f'artifact/sample_imgs/CAM_id_0_{i}/scene_5_000021.jpg')

    with torch.no_grad():
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = feature_extractor.post_process_semantic_segmentation(outputs, [(900, 1600),])[0]
    
    print(f"logits {type(logits)}: {logits.shape}")
    torch.save(logits, f'artifact/CAM_id_0_{i}+scene_5_000021+semseg.pth')

