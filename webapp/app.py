import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import timm

# --- Configuration ---
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", 
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", 
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", 
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", 
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", 
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", 
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", 
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", 
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", 
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", 
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", 
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", 
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", 
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Ensure paths match exactly where your .pth files are saved
RESNET_WEIGHTS_PATH = "resnet18_teacher.pth"
VIT_WEIGHTS_PATH = "vit_tiny.pth"
DEIT_WEIGHTS_PATH = "deit_tiny.pth"

# --- Preprocessing Pipeline ---
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
])

# --- Model Loading ---
def load_models():
    # ResNet18 Teacher using modern weights parameter instead of pretrained
    resnet_model = models.resnet18(weights=None)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 100)
    try:
        resnet_model.load_state_dict(torch.load(RESNET_WEIGHTS_PATH, map_location="cpu"))
    except FileNotFoundError:
        print(f"Warning: {RESNET_WEIGHTS_PATH} not found. Using initialized weights.")
    resnet_model.eval()
    
    # ViT Tiny using standard timm factory
    vit_model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=100)
    try:
        vit_model.load_state_dict(torch.load(VIT_WEIGHTS_PATH, map_location="cpu"))
    except FileNotFoundError:
        print(f"Warning: {VIT_WEIGHTS_PATH} not found. Using initialized weights.")
    vit_model.eval()
    
    # DeiT Tiny using standard timm factory
    deit_model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=100)
    try:
        deit_model.load_state_dict(torch.load(DEIT_WEIGHTS_PATH, map_location="cpu"))
    except FileNotFoundError:
        print(f"Warning: {DEIT_WEIGHTS_PATH} not found. Using initialized weights.")
    deit_model.eval()
    
    return resnet_model, vit_model, deit_model

resnet_teacher, vit_tiny, deit_tiny = load_models()

# --- Visualization Helpers ---
def generate_gradcam(model, input_tensor, original_image):
    features = []
    gradients = []
    
    def hook_f(module, input, output):
        features.append(output)
        
    def hook_b(module, grad_in, grad_out):
        gradients.append(grad_out[0])
        
    target_layer = model.layer4[-1]
    
    # Using modern hook registrations
    h_f = target_layer.register_forward_hook(hook_f)
    h_b = target_layer.register_full_backward_hook(hook_b)
    
    output = model(input_tensor)
    idx = output.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, idx].backward(retain_graph=True)
    
    h_f.remove()
    h_b.remove()
    
    with torch.no_grad():
        fmap = features[0][0]
        grads = gradients[0][0]
        weights = torch.mean(grads, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * fmap, dim=0)
        cam = torch.relu(cam)
        
        if cam.max() != 0:
            cam = cam - cam.min()
            cam = cam / cam.max()
            
        cam_np = cam.cpu().numpy()
        
    heatmap = cv2.resize(cam_np, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(original_image, 0.6, heatmap_img, 0.4, 0)
    return overlay

def generate_vit_attention(model, input_tensor, original_image):
    # Insert your specific notebook attention rollout logic here
    # The models are passed directly, so you can hook into their blocks
    
    with torch.no_grad():
        _ = model(input_tensor)
        
    # Placeholder layout to ensure the UI works before you inject your logic
    heatmap = np.zeros((224, 224), dtype=np.uint8)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(original_image, 0.6, heatmap_img, 0.4, 0)
    return overlay

# --- Inference Logic ---
def run_comparison(input_image):
    if input_image is None:
        return None, "", "", "", None, None, None
        
    display_orig = cv2.resize(input_image, (224, 224))
    tensor_img = transform_pipeline(display_orig).unsqueeze(0)
    
    # ResNet18
    with torch.set_grad_enabled(True):
        resnet_tensor = tensor_img.clone()
        resnet_tensor.requires_grad = True
        
        resnet_output = resnet_teacher(resnet_tensor)
        resnet_probs = torch.softmax(resnet_output, dim=1)[0]
        resnet_idx = resnet_probs.argmax().item()
        resnet_conf = resnet_probs[resnet_idx].item()
        resnet_label = f"{CIFAR100_CLASSES[resnet_idx]} ({resnet_conf:.2%})"
        
        resnet_vis = generate_gradcam(resnet_teacher, resnet_tensor, display_orig)
        
    # ViT Tiny
    with torch.no_grad():
        vit_output = vit_tiny(tensor_img)
        vit_probs = torch.softmax(vit_output, dim=1)[0]
        vit_idx = vit_probs.argmax().item()
        vit_conf = vit_probs[vit_idx].item()
        vit_label = f"{CIFAR100_CLASSES[vit_idx]} ({vit_conf:.2%})"
        
        vit_vis = generate_vit_attention(vit_tiny, tensor_img, display_orig)
        
    # DeiT Tiny
    with torch.no_grad():
        deit_output = deit_tiny(tensor_img)
        deit_probs = torch.softmax(deit_output, dim=1)[0]
        deit_idx = deit_probs.argmax().item()
        deit_conf = deit_probs[deit_idx].item()
        deit_label = f"{CIFAR100_CLASSES[deit_idx]} ({deit_conf:.2%})"
        
        deit_vis = generate_vit_attention(deit_tiny, tensor_img, display_orig)
        
    return display_orig, resnet_label, vit_label, deit_label, resnet_vis, vit_vis, deit_vis

# --- Interface Layout ---
with gr.Blocks(title="CIFAR-100 Model Comparison") as demo:
    gr.Markdown("# CIFAR-100 Model Comparison")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Input Image")
            submit_btn = gr.Button("Run Comparison", variant="primary")
        
        with gr.Column():
            output_orig = gr.Image(label="Original Image (224x224)")
            
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ResNet18 (Teacher)")
            resnet_pred = gr.Textbox(label="Prediction & Confidence")
            resnet_heatmap = gr.Image(label="Grad-CAM Overlay")
            
        with gr.Column():
            gr.Markdown("### ViT Tiny (Normal)")
            vit_pred = gr.Textbox(label="Prediction & Confidence")
            vit_heatmap = gr.Image(label="Attention Overlay")
            
        with gr.Column():
            gr.Markdown("### DeiT Tiny (Distilled)")
            deit_pred = gr.Textbox(label="Prediction & Confidence")
            deit_heatmap = gr.Image(label="Attention Overlay")

    submit_btn.click(
        fn=run_comparison,
        inputs=[input_img],
        outputs=[
            output_orig,
            resnet_pred,
            vit_pred,
            deit_pred,
            resnet_heatmap,
            vit_heatmap,
            deit_heatmap
        ]
    )

if __name__ == "__main__":
    demo.launch()