import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import ResNetModel
from transformers import AutoImageProcessor, ResNetForImageClassification, ViTImageProcessor, ViTModel
from transformers import BertModel
from sentence_transformers import SentenceTransformer
from torchvision.models import resnet50
from transformers import logging
import torch
from transformers import BertModel
from transformers import ViTModel
logging.set_verbosity_warning()

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class MultimodalModel(torch.nn.Module):
    def __init__(self, TextModel, ImageModel):
        torch.nn.Module.__init__(self)
        # Load the BERT model from the local directory

        self.bert = BertModel.from_pretrained(
            '/model/bert',      # 确保这个路径下有 config.json 和权重文件
            local_files_only=True           # 关键：不从网络下载，只用本地
        )

        self.vit = ViTModel.from_pretrained(
            '/model/vit',
            local_files_only=True
        )
        self.image_pool = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.3),
            nn.Tanh()
        )
        self.text_pool = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.3),
            nn.Tanh()
        )

        self.classifier_text = nn.Sequential(
            nn.Linear(in_features=768, out_features=256),
            nn.Dropout(0.3),
            nn.Linear(in_features=256, out_features=1),
            nn.Tanh()
        )

        self.classifier_image = nn.Sequential(
            nn.Linear(in_features=768, out_features=256),
            nn.Dropout(0.3),
            nn.Linear(in_features=256, out_features=1),
            nn.Tanh()
        )

        self.classifier_all = nn.Linear(in_features=768 * 3, out_features=1)


    def forward(self, image_input = None, text_input = None, senti_input = None):
        if (image_input is not None) and (text_input is not None) and (senti_input is not None):

            """Extract text features"""
            text_features = self.bert(**text_input)
            text_hidden_state = text_features.last_hidden_state[:, 0, :]

            """Extract sentiment features"""
            senti_features = self.bert(**senti_input)
            senti_hidden_state = senti_features.last_hidden_state[:, 0, :]

            """Extract image features"""
            image_features = self.vit(**image_input).last_hidden_state
            image_hidden_state, _ = image_features.max(1)

            """Concatenate text and image features to obtain joint representation"""
            image_text_hidden_state = torch.cat([image_hidden_state, text_hidden_state, senti_hidden_state], 1)

            """Perform classification using the concatenated vector"""
            out = self.classifier_all(image_text_hidden_state).squeeze(1)
            out = torch.sigmoid(out)
            return out


        elif image_input is None:
            """text only"""
            assert(text_input is not None)

            text_features = self.bert(**text_input)
            text_hidden_state = text_features.last_hidden_state[:, 0, :]

            # out = self.classifier_single(text_hidden_state).squeeze(1)
            out = self.classifier_text(text_hidden_state).squeeze(1)
            out = torch.sigmoid(out)
            return out


        elif text_input is None:
            """image only"""
            assert(image_input is not None)

            image_features = self.vit(**image_input).last_hidden_state

            image_pooled_output, _ = image_features.max(1)

            out = self.classifier_image(image_pooled_output).squeeze(1)
            out = torch.sigmoid(out)
            return out
