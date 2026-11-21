"""
Model definitions and inference helpers for the jewellery analyser.

Two models share the same idea: a VGG-16 convolutional encoder followed by a GRU.

* VGG16_GRU_Classifier - reads the 7x7 VGG-16 feature map as a sequence of 49
  spatial vectors and classifies the item as Earring / Necklace.
* EncoderCNN + DecoderRNN - image-captioning encoder-decoder. The encoder turns
  the image into a 512-d vector that initialises the GRU decoder, which then
  generates the description word by word (greedy decoding).

The saved .pth files contain the full VGG-16 convolutional weights, so the
backbones are built with ``weights=None`` and no ImageNet download is needed
at start-up.
"""

import pickle

import torch
import torch.nn as nn
from torchvision import models, transforms

# Hyper-parameters used during training (see notebooks/03_captioning_vgg16_gru_training.ipynb)
EMBED_SIZE = 256
HIDDEN_SIZE = 512
CLASSIFIER_HIDDEN = 256
CLASS_NAMES = ["Earring", "Necklace"]

# Same preprocessing as training: resize, tensor conversion, ImageNet normalisation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VGG16_GRU_Classifier(nn.Module):
    def __init__(self, num_classes=2, hidden_size=CLASSIFIER_HIDDEN, gru_layers=1):
        super().__init__()
        self.features = models.vgg16(weights=None).features
        for param in self.features.parameters():
            param.requires_grad = False
        self.gru = nn.GRU(input_size=512, hidden_size=hidden_size,
                          num_layers=gru_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.features(x)                         # (B, 512, 7, 7)
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(0, 2, 1)     # (B, 49, 512)
        _, hidden = self.gru(x)
        return self.fc(hidden.squeeze(0))


class EncoderCNN(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.features = models.vgg16(weights=None).features
        for param in self.features.parameters():
            param.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, hidden_size)

    def forward(self, images):
        x = self.pool(self.features(images))
        return self.fc(x.view(x.size(0), -1))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embedded = self.embed(captions)
        outputs, _ = self.gru(embedded, features.unsqueeze(0))
        return self.fc(outputs)


# ---------------------------------------------------------------------------
# Vocabulary loading
# ---------------------------------------------------------------------------
class Vocabulary:
    """Minimal stand-in for the Vocabulary class pickled in the training notebook."""

    def __init__(self, freq_threshold=1):
        self.itos, self.stoi, self.freq_threshold = {}, {}, freq_threshold

    def __len__(self):
        return len(self.itos)


class _VocabUnpickler(pickle.Unpickler):
    # The notebook pickled the class as __main__.Vocabulary; map it to ours.
    def find_class(self, module, name):
        if name == "Vocabulary":
            return Vocabulary
        return super().find_class(module, name)


def load_vocab(path):
    with open(path, "rb") as f:
        return _VocabUnpickler(f).load()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
_SKIP = {"<PAD>", "<START>", "<END>", "<UNK>", ""}
_PUNCT = [",", ".", "!", "?", ";", ":"]


def generate_caption(image_tensor, encoder, decoder, vocab, max_length=30):
    """Greedy decoding. ``image_tensor`` is an already-transformed (1, 3, 224, 224) tensor."""
    start_token, end_token = vocab.stoi["<START>"], vocab.stoi["<END>"]
    device = image_tensor.device
    with torch.no_grad():
        features = encoder(image_tensor)
        caption = [start_token]
        for _ in range(max_length):
            inputs = torch.tensor(caption, device=device).unsqueeze(0)
            predicted = decoder(features, inputs).argmax(2)[:, -1].item()
            caption.append(predicted)
            if predicted == end_token:
                break

    words = [vocab.itos.get(i, "") for i in caption]
    words = [w for w in words if w not in _SKIP]
    while words and words[0] in _PUNCT:
        words.pop(0)

    sentence = " ".join(words)
    for p in _PUNCT:
        sentence = sentence.replace(" " + p, p)
    sentence = sentence.replace(" 's", "'s").strip()

    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".!?":
            sentence += "."
    return sentence
