from datasets import load_dataset
from transformers import AutoModelForImageClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score


# Load your custom dataset
dataset = load_dataset("path_to_your_dataset")


# Load the pre-trained DINOv2 model
model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-base", num_labels=YOUR_NUM_CLASSES)

model.classifier = torch.nn.Linear(model.config.hidden_size, YOUR_NUM_CLASSES)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
)

trainer.train()

# Perform inference on validation set
predictions = trainer.predict(dataset["val"])


accuracy = accuracy_score(predictions.label_ids, predictions.predictions.argmax(-1))
print(f"Accuracy: {accuracy:.4f}")