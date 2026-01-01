import torch
import torch.nn as nn
from transformers import AutoModel

class FlatHateModel(nn.Module):
    """
    Flat multi-task 모델
    - coarse: 3-class single-label (clean / offensive / hate)
    - fine  : 8개 멀티라벨 (gender, LGBT, age, region, race, religion, socioeconomic, etc)
    - loss = CE_coarse + λ_fine * BCE_fine
    """
    def __init__(
        self,
        plm_name: str,
        num_coarse: int,   # 보통 3
        num_fine: int,     # 보통 8
        lambda_fine: float = 1.0,
        class_weight_coarse: torch.Tensor = None,  # optional: CE용 class weight
        pos_weight_fine: torch.Tensor = None,      # optional: BCE pos_weight
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(plm_name)
        hidden_size = self.encoder.config.hidden_size

        self.head_coarse = nn.Linear(hidden_size, num_coarse)  # (B, 3)
        self.head_fine   = nn.Linear(hidden_size, num_fine)    # (B, 8)

        self.lambda_fine = lambda_fine

        # coarse: 3-class single-label → CrossEntropyLoss
        if class_weight_coarse is not None:
            self.crit_coarse = nn.CrossEntropyLoss(weight=class_weight_coarse)
        else:
            self.crit_coarse = nn.CrossEntropyLoss()

        # fine: multi-label → BCEWithLogitsLoss
        if pos_weight_fine is not None:
            self.crit_fine = nn.BCEWithLogitsLoss(pos_weight=pos_weight_fine)
        else:
            self.crit_fine = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_coarse: torch.Tensor = None,  # [B] (0/1/2)
        label_fine: torch.Tensor = None,    # [B, 8] (0/1)
    ):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0]  # (B, H)

        logits_coarse = self.head_coarse(h_cls)  # (B, num_coarse)
        logits_fine   = self.head_fine(h_cls)    # (B, num_fine)

        result = {
            "logits_coarse": logits_coarse,
            "logits_fine": logits_fine,
        }

        if (label_coarse is not None) and (label_fine is not None):
            # CE는 long [B] 필요
            if label_coarse.dim() != 1:
                label_coarse = label_coarse.view(-1).long()
            else:
                label_coarse = label_coarse.long()

            # BCE는 float [B, F] 필요
            if label_fine.dim() == 1:
                label_fine = label_fine.view(-1, 1)
            label_fine = label_fine.float()

            loss_coarse = self.crit_coarse(logits_coarse, label_coarse)
            loss_fine   = self.crit_fine(logits_fine,   label_fine)
            loss = loss_coarse + self.lambda_fine * loss_fine

            result["loss"] = loss
            result["loss_coarse"] = loss_coarse
            result["loss_fine"]   = loss_fine

        return result