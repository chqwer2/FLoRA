import transformers
import accelerate
import peft

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")

from datasets.utils.logging import set_verbosity_info
set_verbosity_info()
model_checkpoint = "google/vit-base-patch16-224-in21k"

# -----------------------
# Dataset
# -----------------------
from datasets import load_dataset

dataset = load_dataset("food101", split="train[:5000]")
print("loaded dataset:", dataset)

labels = dataset.features["label"].names
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# -----------------------
# Image processor + transforms
# -----------------------
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
print("loaded AutoImageProcessor:", image_processor)

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

def preprocess_train(example_batch):
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

splits = dataset.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# -----------------------
# Utility: print trainable params
# -----------------------
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} "
        f"|| trainable%: {100 * trainable_params / all_param:.4f}"
    )

# -----------------------
# Model
# -----------------------
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

print("Base model params:")
print_trainable_parameters(model)

# -----------------------
# FLoRA (PEFT) setup
# -----------------------
from peft import get_peft_model
from peft import FloraConfig

# ---- YOUR SPECIFIC FLoRA CFG ----
cfg = FloraConfig(
    # LoRA core
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "value"],

    # keep classifier trainable
    modules_to_save=["classifier"],

    # FLoRA extras
    flora_activation="fourier",       # or "relu"/"gelu"/"spline"/"polynomial"/"identity"
    flora_flex_mode="channel",        # stable for ViT shapes
    flora_activation_kwargs={
        # Fourier defaults; change if you want
        "n_terms": 4,
        "init_scale": 0.01,
    },

    # Gate (stability)
    flora_gate_type="sigmoid",        # "none" to disable
    flora_gate_position="after_b",    # "after_a", "after_b", "both"
    flora_gate_init=-6.0,             # starts near-off

    # Debug (off for training)
    flora_debug=False,
    flora_debug_verbose=False,
    flora_debug_forward=False,
    flora_debug_check_nan=True,
)

flora_model = get_peft_model(model, cfg)

print("\nFLoRA-wrapped model params:")
print_trainable_parameters(flora_model)

# If your PEFT fork supports it, this prints a nice summary (optional)
if hasattr(flora_model, "print_trainable_parameters"):
    flora_model.print_trainable_parameters()

# -----------------------
# Training hyperparameters
# -----------------------
from transformers import TrainingArguments, Trainer

model_name = model_checkpoint.split("/")[-1]
batch_size = 128

args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-flora-food101",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    label_names=["labels"],
)

# -----------------------
# Metrics
# -----------------------
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# -----------------------
# Collation
# -----------------------
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# -----------------------
# Train and evaluate
# -----------------------
trainer = Trainer(
    model=flora_model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=image_processor,   # transformers >= 4.40 uses processing_class
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()
print("Train done:", train_results)

eval_results = trainer.evaluate(val_ds)
print("Eval:", eval_results)

