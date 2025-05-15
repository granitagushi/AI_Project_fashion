import gradio as gr
from transformers import pipeline

# 1) Models laden
vit_classifier = pipeline(
    task="image-classification",
    model="Granitagushi/vit-base-fashion"
)
clip_detector = pipeline(
    task="zero-shot-image-classification",
    model="openai/clip-vit-large-patch14"
)

labels = [
    'T - shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# 2) Callback mit str-Casting
def classify_clothing(image):
    vit_results  = vit_classifier(image)
    vit_output   = { str(r["label"]): float(r["score"]) for r in vit_results }

    clip_results = clip_detector(image, candidate_labels=labels)
    clip_output  = { str(r["label"]): float(r["score"]) for r in clip_results }

    return {
        "ViT Classification": vit_output,
        "CLIP Zero-Shot Classification": clip_output
    }

# 3) Beispiele nur, wenn die Dateien ins Repo wandern
example_images = [
    ["example_images/jeans.jpg"],
    ["example_images/kleid.jpg"],
    ["example_images/pullover.jpg"],
    ["example_images/sandale.jpg"],
    ["example_images/tasche.jpg"]
]

# 4) Interface – cache_examples im Konstruktor
iface = gr.Interface(
    fn=classify_clothing,
    inputs=gr.Image(type="pil", label="Upload ein Fashion-MNIST-Bild"),
    outputs=gr.JSON(label="Ergebnisse"),
    title="Fashion MNIST Klassifikation",
    description="Vergleiche dein ViT-Modell mit einem CLIP Zero-Shot-Modell.",
    examples=example_images,    # <— nur, wenn Dateien wirklich da sind
    cache_examples=False
)

if __name__ == "__main__":
    iface.launch(share=True)