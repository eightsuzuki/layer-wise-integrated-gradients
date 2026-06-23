"""
直接計算専用の高速バッチ処理実装

理論文書5.3節の線形性を最大限活用:
- 数値積分不要
- 全Layer×Head×Tokenの一括行列演算
- 大量文書の並列処理
- GPU利用率最大化

従来のIG実装とは完全に分離した専用アーキテクチャ
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DirectComputationBatchProcessor:
    """直接計算専用の高速バッチプロセッサ"""
    
    def __init__(
        self,
        bert_model,
        device: torch.device,
        max_batch_size: int = 32,
        max_sequence_length: int = 128,
        save_cache: bool = True,
        cache_dir: Optional[str] = None,
        model_name: str = "bert-base-uncased",
        split: str = "dev",
        num_steps: int = 32,
        baseline_method: str = "zero",
    ):
        self.bert_model = bert_model
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.num_layers = bert_model.config.num_hidden_layers  # 12
        self.num_heads = bert_model.config.num_attention_heads  # 12
        
        # キャッシュ設定
        self.save_cache = save_cache
        self.cache_dir = cache_dir or "cache/ptb_ig_analysis/samples"
        self.model_name = model_name
        self.split = split
        self.num_steps = num_steps
        self.baseline_method = baseline_method
        self.cache_dir_path = None
        
        if self.save_cache:
            self._setup_cache_directory()
    
    def _setup_cache_directory(self):
        """キャッシュディレクトリを設定"""
        from pathlib import Path
        
        cache_dir_name = f"steps{self.num_steps}_{self.model_name}_maxlen128_v_to_u_direct_baseline_{self.baseline_method}"
        cache_root = Path(self.cache_dir).absolute()
        self.cache_dir_path = cache_root / self.split / "att" / cache_dir_name
        self.cache_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"💾 キャッシュディレクトリ準備: {self.cache_dir_path}")
    
    def _save_batch_results(self, batch_results: List[Dict], start_idx: int):
        """バッチ結果を逐次保存"""
        if not self.save_cache or not self.cache_dir_path:
            return
            
        import json
        
        for i, result in enumerate(batch_results):
            sample_idx = start_idx + i
            sample_file = self.cache_dir_path / f"sample_{sample_idx:05d}.json"
            
            try:
                with open(sample_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"⚠️ サンプル{sample_idx}の保存失敗: {e}")
        
        logger.info(f"💾 バッチ保存完了: サンプル{start_idx}-{start_idx + len(batch_results) - 1}")
        
    def compute_all_contributions_batch(
        self,
        sentences: List[Dict],
        baseline_method: str = "zero",
        progress_callback: Optional[callable] = None,
    ) -> List[Dict]:
        """
        全文書の直接計算を高速バッチ処理で実行
        
        Args:
            sentences: 文書リスト
            baseline_method: ベースライン方法
            progress_callback: 進捗コールバック
            
        Returns:
            List[Dict]: 各文書の貢献度結果
        """
        logger.info(f"🚀 直接計算バッチ処理開始: {len(sentences)}文書")
        start_time = time.time()
        
        results = []
        total_batches = (len(sentences) + self.max_batch_size - 1) // self.max_batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.max_batch_size
            batch_end = min(batch_start + self.max_batch_size, len(sentences))
            batch_sentences = sentences[batch_start:batch_end]
            
            logger.info(f"📦 バッチ {batch_idx + 1}/{total_batches}: {len(batch_sentences)}文書処理中...")
            
            # バッチ処理実行
            batch_results = self._process_batch(batch_sentences, baseline_method)
            results.extend(batch_results)
            
            # 逐次キャッシュ保存
            self._save_batch_results(batch_results, batch_start)
            
            # 進捗報告
            if progress_callback:
                progress = (batch_idx + 1) / total_batches * 100
                progress_callback(f"直接計算バッチ処理: {progress:.1f}%完了")
        
        total_time = time.time() - start_time
        logger.info(f"✅ 直接計算バッチ処理完了: {len(sentences)}文書 ({total_time:.2f}秒)")
        logger.info(f"📊 平均処理速度: {len(sentences)/total_time:.1f}文書/秒")
        
        if self.save_cache and self.cache_dir_path:
            logger.info(f"💾 全キャッシュ保存完了: {len(sentences)}ファイル → {self.cache_dir_path}")
        
        return results
    
    def _process_batch(
        self,
        batch_sentences: List[Dict],
        baseline_method: str,
    ) -> List[Dict]:
        """
        1バッチの文書を処理
        
        Args:
            batch_sentences: バッチ内の文書リスト
            baseline_method: ベースライン方法
            
        Returns:
            List[Dict]: バッチ結果
        """
        # 1. 入力準備
        batch_inputs = self._prepare_batch_inputs(batch_sentences)
        
        # 2. BERT推論（全Layer×Headの情報を一度に取得）
        with torch.no_grad():
            attention_data = self._extract_all_attention_data(batch_inputs)
        
        # 3. 直接計算（全Layer×Head×Tokenを一括処理）
        batch_results = self._compute_batch_contributions(
            attention_data, batch_sentences, baseline_method
        )
        
        return batch_results
    
    def _prepare_batch_inputs(self, batch_sentences: List[Dict]) -> Dict[str, torch.Tensor]:
        """バッチ入力を準備"""
        # tokenizerを取得（アダプターから渡される想定）
        if not hasattr(self, 'tokenizer'):
            # 緊急時の対応: 簡易的なダミートークナイザー
            logger.warning("Tokenizerが設定されていません。ダミーデータを使用します。")
            return self._prepare_dummy_inputs(batch_sentences)
        
        # 文章リストを作成
        texts = []
        for sentence_data in batch_sentences:
            words = sentence_data.get('words', [])
            if not words:
                texts.append("")  # 空文章
            else:
                text = " ".join(words)
                texts.append(text)
        
        # バッチトークン化（全警告を抑制）
        import warnings
        import logging
        
        # 全ての警告とログを一時的に抑制
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # transformersのログレベルを一時的に上げる
            transformers_logger = logging.getLogger("transformers")
            original_level = transformers_logger.level
            transformers_logger.setLevel(logging.ERROR)
            
            try:
                encoded = self.tokenizer(
                    texts,
                    max_length=self.max_sequence_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
            finally:
                # ログレベルを元に戻す
                transformers_logger.setLevel(original_level)
        
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device),
        }
    
    def _prepare_dummy_inputs(self, batch_sentences: List[Dict]) -> Dict[str, torch.Tensor]:
        """ダミー入力を準備（テスト用）"""
        input_ids_list = []
        attention_mask_list = []
        
        for sentence_data in batch_sentences:
            words = sentence_data.get('words', [])
            if not words:
                words = ["dummy"]
                
            # ダミーのトークンID（[CLS] + words + [SEP]）
            input_ids = torch.tensor([101] + list(range(1, len(words) + 1)) + [102])
            
            # パディング
            if len(input_ids) > self.max_sequence_length:
                input_ids = input_ids[:self.max_sequence_length]
            else:
                padding_length = self.max_sequence_length - len(input_ids)
                input_ids = F.pad(input_ids, (0, padding_length), value=0)
            
            attention_mask = (input_ids != 0).long()
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        
        return {
            'input_ids': torch.stack(input_ids_list).to(self.device),
            'attention_mask': torch.stack(attention_mask_list).to(self.device),
        }
    
    def _extract_all_attention_data(self, batch_inputs: Dict[str, torch.Tensor]) -> Dict:
        """
        全Layer×Headのattention情報を一度に抽出
        
        Returns:
            Dict: {
                'attention_weights': [batch, layers, heads, seq_len, seq_len],
                'value_vectors': [batch, layers, seq_len, heads, head_dim],
                'attention_outputs': [batch, layers, seq_len, heads, head_dim],
            }
        """
        batch_size, seq_len = batch_inputs['input_ids'].shape

        # Dropout混入を避けるため、この処理中はevalを強制する
        was_training = self.bert_model.training
        if was_training:
            self.bert_model.eval()

        try:
            # BERT推論（output_attentions=True, output_hidden_states=True）
            outputs = self.bert_model(
                input_ids=batch_inputs['input_ids'],
                attention_mask=batch_inputs['attention_mask'],
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        finally:
            if was_training:
                self.bert_model.train()

        # Attention重み: [batch, layers, heads, seq_len, seq_len]
        attention_weights = torch.stack(outputs.attentions, dim=1)

        # 各層の入力 z^(l): hidden_states[0] は embedding 出力（layer 0入力）
        layer_inputs = torch.stack(outputs.hidden_states[:-1], dim=1)  # [batch, layers, seq, hidden]

        # モデル構造に応じて encoder 層を取得
        if hasattr(self.bert_model, "bert"):
            encoder_layers = self.bert_model.bert.encoder.layer
            config = self.bert_model.bert.config
        else:
            encoder_layers = self.bert_model.encoder.layer
            config = self.bert_model.config

        head_dim = config.hidden_size // self.num_heads

        # Value vectors と Attention outputs を理論式どおりに計算
        value_vectors = []
        attention_outputs = []

        for layer_idx in range(self.num_layers):
            z_l = layer_inputs[:, layer_idx]  # [batch, seq_len, hidden]
            attention_self = encoder_layers[layer_idx].attention.self

            # v_j = W_V z_j + b_V
            value_full = attention_self.value(z_l)  # [batch, seq_len, hidden]
            value_heads = value_full.view(batch_size, seq_len, self.num_heads, head_dim)

            # u_i = Σ_j w_ij v_j
            attn_w = attention_weights[:, layer_idx]  # [batch, heads, seq, seq]
            attn_out = torch.einsum("bhij,bjhd->bihd", attn_w, value_heads)

            value_vectors.append(value_heads)
            attention_outputs.append(attn_out)

        return {
            'attention_weights': attention_weights,
            'value_vectors': torch.stack(value_vectors, dim=1),  # [batch, layers, seq, heads, head_dim]
            'attention_outputs': torch.stack(attention_outputs, dim=1),  # [batch, layers, seq, heads, head_dim]
        }
    
    def _compute_batch_contributions(
        self,
        attention_data: Dict,
        batch_sentences: List[Dict],
        baseline_method: str,
    ) -> List[Dict]:
        """
        バッチ全体の貢献度を一括計算
        
        理論式の直接実装:
        - Zero Baseline: Contribution(v_j → u_i) = w_ij * v_j
        - Self Input Token: Contribution(v_j → u_i) = w_ij * (v_j - v_i)
        - Self Output Token: Contribution(v_j → u_i) = w_ij * (v_j - u_i)
        """
        batch_size = len(batch_sentences)
        attention_weights = attention_data['attention_weights']  # [batch, layers, heads, seq_len, seq_len]
        value_vectors = attention_data['value_vectors']  # [batch, layers, seq_len, heads, head_dim]
        attention_outputs = attention_data['attention_outputs']  # [batch, layers, seq_len, heads, head_dim]
        
        results = []
        
        for batch_idx in range(batch_size):
            sentence_data = batch_sentences[batch_idx]
            sentence_results = {
                'tokens': sentence_data.get('tokens', sentence_data.get('words', [])),
                'words': sentence_data.get('words', []),
                'attns': {},
            }
            
            # 各Layer×Headの貢献度を計算
            for layer_idx in range(self.num_layers):
                layer_results = {}
                
                for head_idx in range(self.num_heads):
                    # 現在のLayer×Headのデータ
                    w_ij = attention_weights[batch_idx, layer_idx, head_idx]  # [seq_len, seq_len]
                    v_j = value_vectors[batch_idx, layer_idx, :, head_idx]  # [seq_len, head_dim]
                    u_i = attention_outputs[batch_idx, layer_idx, :, head_idx]  # [seq_len, head_dim]
                    
                    # 貢献度計算
                    if baseline_method == "zero":
                        # Contribution(v_j → u_i) = w_ij * v_j
                        contributions = torch.einsum('ij,jd->ijd', w_ij, v_j)  # [seq_len, seq_len, head_dim]
                    
                    elif baseline_method == "self_input_token":
                        # Contribution(v_j → u_i) = w_ij * (v_j - v_i)
                        v_diff = v_j.unsqueeze(0) - v_j.unsqueeze(1)  # [seq_len, seq_len, head_dim]
                        contributions = w_ij.unsqueeze(-1) * v_diff  # [seq_len, seq_len, head_dim]
                    
                    else:
                        raise ValueError(f"Unknown baseline_method: {baseline_method}")
                    
                    # L2ノルムでスカラー化
                    ig_values = torch.norm(contributions, p=2, dim=-1)  # [seq_len, seq_len]
                    
                    # 実際の文長を取得
                    actual_length = len(sentence_data.get('tokens', sentence_data.get('words', [])))
                    
                    # パディング部分を除去して実際の文長に切り詰める
                    ig_values_trimmed = ig_values[:actual_length, :actual_length]
                    
                    # CPUに移動してリスト化
                    layer_results[head_idx] = ig_values_trimmed.cpu().tolist()
                
                sentence_results['attns'][layer_idx] = layer_results
            
            results.append(sentence_results)
        
        return results


def compute_direct_batch_fast(
    sentences: List[Dict],
    bert_model,
    device: torch.device,
    baseline_method: str = "zero",
    max_batch_size: int = 32,
    progress_callback: Optional[callable] = None,
    save_cache: bool = True,
    cache_dir: Optional[str] = None,
    model_name: str = "bert-base-uncased",
    split: str = "dev",
    num_steps: int = 32,
) -> Tuple[List[Dict], Optional[str]]:
    """
    直接計算の高速バッチ処理エントリーポイント
    
    Args:
        sentences: 文書リスト
        bert_model: BERTモデル
        device: 計算デバイス
        baseline_method: ベースライン方法
        max_batch_size: 最大バッチサイズ
        progress_callback: 進捗コールバック
        save_cache: キャッシュに保存するか
        cache_dir: キャッシュディレクトリ
        model_name: モデル名
        split: データセット分割
        num_steps: IGステップ数（キャッシュディレクトリ名用）
        
    Returns:
        Tuple[List[Dict], Optional[str]]: 各文書の貢献度結果とキャッシュディレクトリパス
    """
    processor = DirectComputationBatchProcessor(
        bert_model=bert_model,
        device=device,
        max_batch_size=max_batch_size,
        save_cache=save_cache,
        cache_dir=cache_dir,
        model_name=model_name,
        split=split,
        num_steps=num_steps,
        baseline_method=baseline_method,
    )
    
    results = processor.compute_all_contributions_batch(
        sentences=sentences,
        baseline_method=baseline_method,
        progress_callback=progress_callback,
    )
    
    # キャッシュディレクトリパスを返す
    cache_dir_path = str(processor.cache_dir_path) if processor.cache_dir_path else None
    
    return results, cache_dir_path


def save_results_to_cache(
    results: List[Dict],
    baseline_method: str,
    cache_dir: Optional[str] = None,
    model_name: str = "bert-base-uncased",
    split: str = "dev",
    num_steps: int = 32,
) -> str:
    """
    結果をキャッシュに保存
    
    Args:
        results: 計算結果
        baseline_method: ベースライン方法
        cache_dir: ベースキャッシュディレクトリ
        model_name: モデル名
        split: データセット分割
        num_steps: IGステップ数
        
    Returns:
        str: 保存先ディレクトリパス
    """
    import json
    import os
    from pathlib import Path
    
    # キャッシュディレクトリ作成
    if cache_dir is None:
        cache_dir = "cache/ptb_ig_analysis/samples"
    
    cache_dir_name = f"steps{num_steps}_{model_name}_maxlen128_v_to_u_direct_baseline_{baseline_method}"
    full_cache_dir = Path(cache_dir) / split / cache_dir_name
    full_cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 キャッシュ保存開始: {full_cache_dir}")
    
    # 各サンプルを個別ファイルに保存
    for idx, result in enumerate(results):
        sample_file = full_cache_dir / f"sample_{idx:05d}.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ キャッシュ保存完了: {len(results)}ファイル → {full_cache_dir}")
    
    return str(full_cache_dir)
