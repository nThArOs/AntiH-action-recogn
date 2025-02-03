import os

import pandas as pd
import torch
import numpy as np
import pathlib
from transformers import VideoMAEForVideoClassification, AutoImageProcessor, VideoMAEImageProcessor, TrainingArguments, \
    Trainer, AutoConfig, AutoModel
import pytorchvideo.data
import imageio
from IPython.display import Image
import evaluate

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)


def unnormalize_img(img):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def create_gif(video_tensor, filename="sample.gif"):
    """Prepares a GIF from a video tensor.
...
    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
        frames.append(frame_unnormalized)
    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename


def display_gif(video_tensor, gif_name="sample.gif"):
    """Prepares and displays a GIF from a video tensor."""
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, gif_name)
    return Image(filename=gif_filename)


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


dataset_root_path = "antihpert-gre"

dataset_root_path = pathlib.Path(dataset_root_path)

video_count_train = len(list(dataset_root_path.glob("train/*/*.mp4")))
video_count_val = len(list(dataset_root_path.glob("val/*/*.mp4")))
video_count_test = len(list(dataset_root_path.glob("test/*/*.mp4")))

all_video_file_paths = (
        list(dataset_root_path.glob("train/*/*.mp4"))
        + list(dataset_root_path.glob("val/*/*.mp4"))
        + list(dataset_root_path.glob("test/*/*.mp4"))
)

print(all_video_file_paths[:5])

class_labels = sorted({str(path).split("/")[2] for path in all_video_file_paths})
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

model_ckpt = "MCG-NJU/videomae-base"
batch_size = 8  # batch size for training and evaluation

image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

print(f"Unique classes: {list(label2id.keys())}.")

mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps

train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)

train_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=train_transform,
)

val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

val_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "val"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

test_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "test"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

print(train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)


# sample_test_video = "563.mp4"
def run_inference(model, video):
    # (num_frames, num_channels, height, width)
    perumuted_sample_test_video = video.permute(1, 0, 2, 3)
    inputs = {
        "pixel_values": perumuted_sample_test_video.unsqueeze(0),
        "labels": torch.tensor(
            [sample_test_video["label"]]
        ),  # this can be skipped if you don't have labels available.
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return logits

sample_test_video = next(iter(test_dataset))
print((test_dataset))
trained_model = model.from_pretrained("./videomae-base-finetuned-ucf101-subset/checkpoint-29040", config=model.config)
# trained_model = model="videomae-base-finetuned-ucf101-subset"
#logits = run_inference(trained_model, sample_test_video["video"])
#predicted_class_idx = logits.argmax(-1).item()
list_label = list(label2id.keys())

#print(f"Predicted:{model.config.id2label[predicted_class_idx]} /Ground:{list_label[sample_test_video['label']]}")
display_gif(sample_test_video["video"])
score = 0
ope_scoe = 0
total = 0
"""for test_video in iter(test_dataset):
    if test_video["label"] == 4:
        total += 1
        logits = run_inference(trained_model, test_video["video"])
        predicted_class_idx = logits.argmax(-1).item()
        pred_op = model.config.id2label[predicted_class_idx].split("-")
        ground_op = list_label[test_video['label']].split('-')
        #print(test_video)
        print(f"{test_video['video_name']} I Predicted:{model.config.id2label[predicted_class_idx]} /Ground:{list_label[test_video['label']]} - "
              f"{model.config.id2label[predicted_class_idx] == list_label[test_video['label']]} -"
              f"{pred_op[0] == ground_op[0 ]}")
        if model.config.id2label[predicted_class_idx] == list_label[test_video['label']]:
            score +=1
        if pred_op[0] == ground_op[0]:
            ope_scoe += 1"""
actual = []
predicted = []
for test_video in iter(test_dataset):
    total += 1
    logits = run_inference(trained_model, test_video["video"])
    predicted_class_idx = logits.argmax(-1).item()
    pred_op = model.config.id2label[predicted_class_idx].split("-")
    ground_op = list_label[test_video['label']].split('-')
    actual.append(list_label[test_video['label']])
    predicted.append(model.config.id2label[predicted_class_idx])
    #print(test_video)
    print(f"{test_video['video_name']} I Predicted:{model.config.id2label[predicted_class_idx]} / Ground:{list_label[test_video['label']]} - "
          f"{model.config.id2label[predicted_class_idx] == list_label[test_video['label']]} -"
          f"{pred_op[0] == ground_op[0 ]}")
    if model.config.id2label[predicted_class_idx] == list_label[test_video['label']]:
        score +=1
    if pred_op[0] == ground_op[0]:
        ope_scoe += 1
print(f"total ={total} - score = {score} - op score = {ope_scoe}")
conf = {"pred":predicted,"ground":actual}
df=pd.DataFrame(conf)
df.to_csv('result_pred.csv', index=False)