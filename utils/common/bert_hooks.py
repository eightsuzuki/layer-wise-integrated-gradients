# bert_hooks.py
import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertConfig, BertModel
from typing import Dict, Tuple, Optional, List
import lightning as L

# GPU使用可能かどうかをチェック
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------------
# PyTorch Lightningを使用したBERTモデル
# ----------------------------------------------------------------------------
class BertLightningModule(L.LightningModule):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.model_name = model_name
        self.config = BertConfig.from_pretrained(model_name)
        self.config.output_attentions = True
        self.config.output_hidden_states = True
        
        # BERTモデルを初期化
        self.bert = BertModel.from_pretrained(
            model_name, 
            config=self.config,
            attn_implementation="eager"
        )
        
        # 中間出力保存用
        self.outputs = {
            "attn": {},      # Attention出力
            "mlp_inter": {}, # MLP中間出力
            "mlp_out": {},   # MLP最終出力
            "ln": {},        # LayerNorm出力
            "next_val": {},  # 次層Value
            "qkv": {"q":{}, "k":{}, "v":{}}, # QKV
            "attn_weights": {}, # Attention重み
            "head_outputs": {} # 各ヘッドの出力
        }
        
        # フック登録
        self._register_hooks()
    
    def _register_hooks(self):
        """フックを登録"""
        for idx, layer in enumerate(self.bert.encoder.layer):
            # Attention系
            layer.attention.self.query.register_forward_hook(self._hook_qkv("q", idx))
            layer.attention.self.key.register_forward_hook(self._hook_qkv("k", idx))
            layer.attention.self.value.register_forward_hook(self._hook_qkv("v", idx))
            layer.attention.self.register_forward_hook(self._hook_attn_weights(idx))
            layer.attention.self.register_forward_hook(self._hook_next_value(idx))
            layer.attention.output.register_forward_hook(self._hook_head_outputs(idx))
            
            # MLP系
            layer.output.LayerNorm.register_forward_hook(self._hook_store("attn", idx))
            layer.intermediate.dense.register_forward_hook(self._hook_store("mlp_inter", idx))
            layer.output.dense.register_forward_hook(self._hook_store("mlp_out", idx))
            layer.output.LayerNorm.register_forward_hook(self._hook_store("ln", idx))

    def _hook_store(self, key: str, idx: int):
        def hook(module, inp, out):
            val = inp[0] if isinstance(inp, tuple) else inp
            self.outputs[key][idx] = val.detach()
        return hook
    
    def _hook_qkv(self, kind: str, idx: int):
        def hook(module, inp, out):
            self.outputs["qkv"][kind][idx] = out.detach()
        return hook
    
    def _hook_attn_weights(self, idx: int):
        def hook(module, inp, out):
            try:
                q = self.outputs["qkv"]["q"][idx]
                k = self.outputs["qkv"]["k"][idx]
                batch_size, seq_len, _ = q.shape
                num_heads = module.num_attention_heads
                head_dim = module.attention_head_size
                q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1,2)
                k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1,2)
                scores = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(head_dim)
                self.outputs["attn_weights"][idx] = F.softmax(scores, dim=-1).detach()
            except Exception as e:
                print(f"Error in attention weights hook layer {idx}: {e}")
        return hook
    
    def _hook_next_value(self, idx):
        def hook(module, inp, out):
            try:
                val = out[0] if isinstance(out, tuple) else out
                self.outputs["next_val"][idx] = val.detach()
            except Exception as e:
                print(f"Error in next value hook layer {idx}: {e}")
        return hook
    
    def _hook_head_outputs(self, idx):
        def hook(module, inp, out):
            try:
                # 各ヘッドの出力を保存
                self.outputs["head_outputs"][idx] = out[0].detach()
            except Exception as e:
                print(f"Error in head outputs hook layer {idx}: {e}")
        return hook

    def forward(self, **inputs):
        """フォワードパス"""
        return self.bert(**inputs)

    def get_output(self, key: str, layer_idx: int) -> Optional[torch.Tensor]:
        return self.outputs[key].get(layer_idx)
    
    def get_qkv(self, kind: str, layer_idx: int) -> Optional[torch.Tensor]:
        return self.outputs["qkv"][kind].get(layer_idx)
    
    def get_value(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.outputs["qkv"]["v"].get(layer_idx)

    def get_head_separated_attention_output(self, layer_idx: int, target_head: int) -> Optional[torch.Tensor]:
        """
        特定のヘッドのAttention出力のみを取得
        
        Args:
            layer_idx: 対象レイヤー
            target_head: 対象ヘッド
            
        Returns:
            Optional[torch.Tensor]: 特定ヘッドのAttention出力 [batch, seq_len, head_dim]
        """
        # Attention出力を取得
        attention_output = self.outputs["head_outputs"].get(layer_idx)
        if attention_output is None:
            return None
            
        # ヘッド分割
        batch_size, seq_len, hidden_size = attention_output.shape
        num_heads = self.config.num_attention_heads
        head_dim = hidden_size // num_heads
        
        # [batch, seq_len, hidden_size] → [batch, num_heads, seq_len, head_dim]
        attention_heads = attention_output.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # 特定ヘッドのみを取得
        target_head_output = attention_heads[:, target_head, :, :]  # [batch, seq_len, head_dim]
        
        return target_head_output

    def get_all_heads_attention_output(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        全ヘッドのAttention出力を取得
        
        Args:
            layer_idx: 対象レイヤー
            
        Returns:
            Optional[torch.Tensor]: 全ヘッドのAttention出力 [batch, num_heads, seq_len, head_dim]
        """
        attention_output = self.outputs["head_outputs"].get(layer_idx)
        if attention_output is None:
            return None
            
        # ヘッド分割
        batch_size, seq_len, hidden_size = attention_output.shape
        num_heads = self.config.num_attention_heads
        head_dim = hidden_size // num_heads
        
        # [batch, seq_len, hidden_size] → [batch, num_heads, seq_len, head_dim]
        attention_heads = attention_output.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        return attention_heads

    def get_mlp(self, layer_idx: int):
        return self.outputs["mlp_inter"].get(layer_idx), self.outputs["mlp_out"].get(layer_idx)

    def get_x(self, layer_idx: int):
        return self.outputs["attn"].get(layer_idx)

# ----------------------------------------------------------------------------
# 1. BERT の MLP 部分にフックを登録して中間・最終出力を取得するクラス
# ----------------------------------------------------------------------------
class BertWithMLPHooks(BertModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.mlp_intermediate = {}  # プリ活性化 z = W1 x + b1
        self.mlp_output = {}        # ポスト活性化 y = W2 GELU(z) + b2
        self.attn_output = {}       # Attention 出力 = x
        self.next_attn_value = {}
        for idx, layer in enumerate(self.encoder.layer):
            layer.output.LayerNorm.register_forward_hook(self._save_attn_output_hook(idx))
            layer.intermediate.dense.register_forward_hook(self._save_inter_hook(idx))
            layer.output.dense.register_forward_hook(self._save_output_hook(idx))
            layer.attention.self.register_forward_hook(self._save_next_value_hook(idx))

    def _save_attn_output_hook(self, layer_idx):
        def hook(module, inp, out):
            # Attention + 出力線形後に residual を加えたベクトル (x) が LayerNorm に入る
            # inp[0] がそのテンソル
            self.attn_output[layer_idx] = inp[0].detach()
        return hook

    def _save_next_value_hook(self, layer_idx):
        def hook(module, inp, out):
            # out は (context_layer, attn_probs) または (context_layer,) などのタプル
            ctx = out[0] if isinstance(out, tuple) else out
            # ctx: (batch, heads, seq_len, head_dim)
            self.next_attn_value[layer_idx] = ctx.detach()
        return hook

    def get_value(self, layer_idx: int) -> Optional[torch.Tensor]:
        """指定されたレイヤーのValueテンソルを取得"""
        return self.next_attn_value.get(layer_idx)

    def _save_inter_hook(self, layer_idx: int):
        def hook(module, inp, out):
            self.mlp_intermediate[layer_idx] = out.detach()
        return hook

    def _save_output_hook(self, layer_idx: int):
        def hook(module, inp, out):
            self.mlp_output[layer_idx] = out.detach()
        return hook

    def get_mlp(self, layer_idx: int):
        return self.mlp_intermediate.get(layer_idx), self.mlp_output.get(layer_idx)

    def get_x(self, layer_idx: int):
        return self.attn_output.get(layer_idx)

# ----------------------------------------------------------------------------
# Attentionフック付きモデル
# ----------------------------------------------------------------------------
class BertWithHooks(BertModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        
        # 中間出力保存用
        self.outputs = {
            "attn": {},      # Attention出力
            "mlp_inter": {}, # MLP中間出力
            "mlp_out": {},   # MLP最終出力
            "ln": {},        # LayerNorm出力
            "next_val": {},  # 次層Value
            "qkv": {"q":{}, "k":{}, "v":{}}, # QKV
            "attn_weights": {}, # Attention重み
            "head_outputs": {} # 各ヘッドの出力
        }
        
        # フック登録
        for idx, layer in enumerate(self.encoder.layer):
            # Attention系
            layer.attention.self.query.register_forward_hook(self._hook_qkv("q", idx))
            layer.attention.self.key.register_forward_hook(self._hook_qkv("k", idx))
            layer.attention.self.value.register_forward_hook(self._hook_qkv("v", idx))
            layer.attention.self.register_forward_hook(self._hook_attn_weights(idx))
            layer.attention.self.register_forward_hook(self._hook_next_value(idx))
            layer.attention.output.register_forward_hook(self._hook_head_outputs(idx))
            
            # MLP系
            layer.output.LayerNorm.register_forward_hook(self._hook_store("attn", idx))
            layer.intermediate.dense.register_forward_hook(self._hook_store("mlp_inter", idx))
            layer.output.dense.register_forward_hook(self._hook_store("mlp_out", idx))
            layer.output.LayerNorm.register_forward_hook(self._hook_store("ln", idx))

    def _hook_store(self, key: str, idx: int):
        def hook(module, inp, out):
            val = inp[0] if isinstance(inp, tuple) else inp
            self.outputs[key][idx] = val.detach()
        return hook
    
    def _hook_qkv(self, kind: str, idx: int):
        def hook(module, inp, out):
            self.outputs["qkv"][kind][idx] = out.detach()
        return hook
    
    def _hook_attn_weights(self, idx: int):
        def hook(module, inp, out):
            try:
                q = self.outputs["qkv"]["q"][idx]
                k = self.outputs["qkv"]["k"][idx]
                batch_size, seq_len, _ = q.shape
                num_heads = module.num_attention_heads
                head_dim = module.attention_head_size
                q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1,2)
                k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1,2)
                scores = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(head_dim)
                self.outputs["attn_weights"][idx] = F.softmax(scores, dim=-1).detach()
            except Exception as e:
                print(f"Error in attention weights hook layer {idx}: {e}")
        return hook
    
    def _hook_next_value(self, idx):
        def hook(module, inp, out):
            try:
                val = out[0] if isinstance(out, tuple) else out
                self.outputs["next_val"][idx] = val.detach()
            except Exception as e:
                print(f"Error in next value hook layer {idx}: {e}")
        return hook
    
    def _hook_head_outputs(self, idx):
        def hook(module, inp, out):
            try:
                # 各ヘッドの出力を保存
                self.outputs["head_outputs"][idx] = out[0].detach()
            except Exception as e:
                print(f"Error in head outputs hook layer {idx}: {e}")
        return hook

    # ゲッター
    def get_output(self, key: str, layer_idx: int) -> Optional[torch.Tensor]:
        return self.outputs[key].get(layer_idx)
    
    def get_qkv(self, kind: str, layer_idx: int) -> Optional[torch.Tensor]:
        return self.outputs["qkv"][kind].get(layer_idx)
    
    def get_value(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.outputs["qkv"]["v"].get(layer_idx)

# ----------------------------------------------------------------------------
# モデル & トークナイザー読み込み
# ----------------------------------------------------------------------------
def load_mlp_model(model_name: str = "bert-base-uncased") -> BertWithMLPHooks:
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    model = BertWithMLPHooks.from_pretrained(
        model_name, 
        config=config,
        attn_implementation="eager"  # SDPAとの競合を回避
    )
    model.eval()
    model.to(DEVICE)  # GPUに移動
    return model

def load_attn_model(model_name: str = "bert-base-uncased") -> BertWithHooks:
    config = BertConfig.from_pretrained(model_name)
    config.output_attentions = True
    model = BertWithHooks.from_pretrained(
        model_name, 
        config=config,
        attn_implementation="eager"  # SDPAとの競合を回避
    )
    model.eval()
    model.to(DEVICE)  # GPUに移動
    return model

def load_lightning_model(model_name: str = "bert-base-uncased") -> BertLightningModule:
    """PyTorch Lightningを使用したBERTモデルを読み込み"""
    model = BertLightningModule(model_name)
    model.eval()
    model.to(DEVICE)  # GPUに移動
    return model