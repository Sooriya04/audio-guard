"""
model.py — Wav2Vec2-based binary classifier for deepfake audio detection.

Architecture
────────────
  Wav2Vec2 (frozen CNN feature extractor, trainable transformer layers)
  → mean-pool hidden states
  → LayerNorm → Dense(768/1024 → 256) → GELU → Dropout → Dense(256 → 2)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

import config as cfg

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Classification Head
# ──────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Two-layer MLP head on top of mean-pooled Wav2Vec2 hidden states.

    Parameters
    ----------
    hidden_size : int  — Wav2Vec2 hidden dimension (768 for base, 1024 for large)
    num_labels  : int  — number of output classes (2 for real/fake)
    dropout     : float
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = cfg.NUM_LABELS,
        dropout: float = cfg.CLASSIFIER_DROPOUT,
    ):
        super().__init__()
        mid = max(256, hidden_size // 4)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────
#  Full Detector Model
# ──────────────────────────────────────────────────────────

class DeepfakeAudioDetector(Wav2Vec2PreTrainedModel):
    """
    Wav2Vec2 encoder with a custom classification head for real/fake detection.

    Inherits from Wav2Vec2PreTrainedModel to get all HuggingFace conveniences
    (save_pretrained, from_pretrained, gradient checkpointing, etc.).
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config     = config
        self.wav2vec2   = Wav2Vec2Model(config)
        self.classifier = ClassificationHead(
            hidden_size=config.hidden_size,
            num_labels=config.num_labels if hasattr(config, "num_labels") else cfg.NUM_LABELS,
            dropout=cfg.CLASSIFIER_DROPOUT,
        )
        self.init_weights()

    @property
    def all_tied_weights_keys(self):
        """Compatibility hack for transformers 5.x — expects a dict."""
        val = getattr(self, "_tied_weights_keys", {})
        return val if val is not None else {}

    # ── pooling ────────────────────────────────────────────

    def _mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Weighted mean-pool over time:
          - mask out padding positions
          - return shape (batch, hidden_size)
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        # Wav2Vec2 downsamples the sequence length; re-scale the mask.
        # hidden_states: (B, T', H)  attention_mask: (B, T_raw)
        T_enc = hidden_states.shape[1]
        T_raw = attention_mask.shape[1]
        if T_enc != T_raw:
            # nearest-neighbour rescaling along time axis
            mask_float = attention_mask.float().unsqueeze(1)          # (B, 1, T_raw)
            mask_float = torch.nn.functional.interpolate(
                mask_float, size=T_enc, mode="nearest"
            ).squeeze(1)                                               # (B, T_enc)
        else:
            mask_float = attention_mask.float()

        mask_expanded = mask_float.unsqueeze(-1)                       # (B, T_enc, 1)
        sum_hidden    = (hidden_states * mask_expanded).sum(dim=1)
        count         = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / count

    # ── forward ────────────────────────────────────────────

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> SequenceClassifierOutput:

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state          # (B, T, H)
        pooled        = self._mean_pool(hidden_states, attention_mask)
        logits        = self.classifier(pooled)            # (B, num_labels)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss    = loss_fn(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )


# ──────────────────────────────────────────────────────────
#  Builder
# ──────────────────────────────────────────────────────────

def build_model(
    model_name: str = cfg.MODEL_NAME,
    num_labels: int = cfg.NUM_LABELS,
    freeze_feature_extractor: bool = cfg.FREEZE_FEATURE_EXTRACTOR,
) -> DeepfakeAudioDetector:
    """
    Load a pre-trained Wav2Vec2 config + weights, attach our head,
    and optionally freeze the CNN feature extractor.
    """
    from transformers import Wav2Vec2Config

    w2v_config = Wav2Vec2Config.from_pretrained(model_name)
    w2v_config.num_labels = num_labels
    # Reduce hidden dropout to avoid over-regularisation
    w2v_config.hidden_dropout      = cfg.HIDDEN_DROPOUT
    w2v_config.attention_dropout   = cfg.HIDDEN_DROPOUT
    w2v_config.feat_proj_dropout   = 0.0
    w2v_config.mask_time_prob      = 0.05   # light masking during fine-tune

    model = DeepfakeAudioDetector.from_pretrained(
        model_name,
        config=w2v_config,
        ignore_mismatched_sizes=True,
    )

    if freeze_feature_extractor:
        # Freeze the CNN feature extractor (first ~10 conv layers)
        model.wav2vec2.feature_extractor._freeze_parameters()
        log.info("CNN feature extractor frozen.")

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        f"Model: {model_name}  |  "
        f"total params: {n_total/1e6:.1f}M  |  "
        f"trainable: {n_trainable/1e6:.1f}M"
    )
    return model


def load_model(checkpoint_dir: str, device: Optional[torch.device] = None) -> DeepfakeAudioDetector:
    """Load a saved model from a HuggingFace-style checkpoint directory."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeAudioDetector.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()
    return model
