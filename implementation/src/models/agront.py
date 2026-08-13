"""AgroNT（1B DNA 大模型）+ LoRA 微调分类/回归头。

加载: InstaDeepAI/agro-nucleotide-transformer-1b（HuggingFace，6-mer tokenizer）
说明: 主干经 PEFT LoRA 冻结训练；隐藏维度从 config 读取，不硬编码。
      target_modules 若与实际模块名不符，运行后打印 backbone.named_modules() 核对。
"""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger("plantdl")


class AgroNTClassifier(nn.Module):
    def __init__(self, model_id, num_classes=1, task="binary",
                 lora_r=16, lora_alpha=32, lora_dropout=0.1,
                 target_modules=("q_lin", "k_lin", "v_lin", "out_lin"),
                 max_seq_len=None):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer

        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        base = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        hidden = getattr(base.config, "hidden_size", 1024)
        logger.info("AgroNT hidden_size=%d 参数量=%d", hidden, base.num_parameters())
        # 梯度检查点：大幅降低激活显存（40 层 1B 模型必需）。
        # 注意：AutoModel 实际加载的是 transformers 内置的 ESM（非 remote code），
        # 需用官方 gradient_checkpointing_enable()（同时设置 _gradient_checkpointing_func）。
        try:
            base.gradient_checkpointing_enable()
            logger.info("已开启梯度检查点")
        except Exception as e:
            logger.warning("梯度检查点开启失败: %s", e)

        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            # 用 FEATURE_EXTRACTION 而非 SEQ_CLS：自带头做分类，
            # 避免 PeftModelForSequenceClassification 向 EsmModel 传 labels 参数报错
            task_type="FEATURE_EXTRACTION",
        )
        self.backbone = get_peft_model(base, lora_cfg)
        self.backbone.print_trainable_parameters()
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
        self.max_seq_len = max_seq_len

    def forward(self, seqs):
        """seqs: List[str] 或 batch 张量（tokenized）。
        文本输入 -> 自动 tokenize；张量输入 -> 直接前向（用于归因）。"""
        if isinstance(seqs, (list, tuple)) or isinstance(seqs, str):
            seqs = [seqs] if isinstance(seqs, str) else list(seqs)
            # N 富集序列会 tokenize 出远超 bp/6 的 token 数（N 用独立 token），
            # 超过 max_position_embeddings(1026) 会触发 device-side assert。
            # 用 token 级上限截断（6000bp ≈ 1001 tokens，上限 1020 留余量）。
            token_cap = min(1020, (self.max_seq_len or 6000) // 6 + 2)
            toks = self.tokenizer(seqs, return_tensors="pt",
                                  padding=True, truncation=True,
                                  max_length=token_cap)
            input_ids = toks["input_ids"].to(self.backbone.device)
            attn = toks["attention_mask"].to(self.backbone.device)
        else:
            input_ids, attn = seqs, None

        out = self.backbone(input_ids=input_ids, attention_mask=attn)
        hs = out.last_hidden_state
        # mean-pool（忽略 pad）；若用 CLS 方式可改为 hs[:, 0]
        if attn is not None:
            mask = attn.unsqueeze(-1).float()
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hs.mean(1)
        return self.head(pooled)

    def print_modules(self):
        print(">>> backbone 模块名（供核对 target_modules）:")
        for n, _ in self.backbone.named_modules():
            if "lin" in n or "attn" in n or "self" in n:
                print("   ", n)


def build_agront(cfg: dict, num_classes=1, task="binary") -> AgroNTClassifier:
    m = cfg["model"]["agront"]
    return AgroNTClassifier(
        model_id=m["model_id"],
        num_classes=num_classes,
        task=task,
        lora_r=m["lora_r"],
        lora_alpha=m["lora_alpha"],
        lora_dropout=m["lora_dropout"],
        target_modules=tuple(m["target_modules"]),
        max_seq_len=m.get("max_bp"),
    )
