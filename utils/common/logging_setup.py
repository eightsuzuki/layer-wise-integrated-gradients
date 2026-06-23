"""
共通ログ設定モジュール

Streamlit版とCLI版の両方で使用できる統一ログ設定
"""

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}

DISABLE_EMOJI = _env_flag("PTB_DISABLE_LOG_EMOJI", True)  # デフォルトで絵文字を無効化（文字化け対策）

def _sanitize_text(text: str) -> str:
    if not DISABLE_EMOJI or not isinstance(text, str):
        return text
    return text.encode("ascii", errors="ignore").decode("ascii", errors="ignore")

class AsciiOnlyFilter(logging.Filter):
    """Optionally strip non-ASCII characters (emoji) from log messages"""
    def filter(self, record):
        if DISABLE_EMOJI and isinstance(record.msg, str):
            record.msg = _sanitize_text(record.msg)
            if record.args:
                record.args = tuple(
                    _sanitize_text(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )
        return True


# ANSIカラーコード（コンソール用）
class Colors:
    """ANSIカラーコード"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # GPUごとの色（8色）
    GPU_COLORS = [
        '\033[91m',  # GPU0: 赤
        '\033[92m',  # GPU1: 緑
        '\033[94m',  # GPU2: 青
        '\033[95m',  # GPU3: マゼンタ
        '\033[96m',  # GPU4: シアン
        '\033[93m',  # GPU5: 黄
        '\033[97m',  # GPU6: 白
        '\033[90m',  # GPU7: グレー
    ]
    
    @staticmethod
    def get_gpu_color(gpu_id: int) -> str:
        """GPU IDに対応する色を取得"""
        return Colors.GPU_COLORS[gpu_id % len(Colors.GPU_COLORS)]
    
    @staticmethod
    def strip_colors(text: str) -> str:
        """ANSIカラーコードを削除（ログファイル用）"""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)


class JSTFormatter(logging.Formatter):
    """日本時間（JST）でログをフォーマットするフォーマッター（GPU色分け対応）"""

    JST = timezone(timedelta(hours=9))

    def __init__(self, fmt=None, datefmt=None, enable_colors=True):
        """フォーマッターを初期化
        
        Args:
            fmt: フォーマット文字列
            datefmt: 日時フォーマット文字列
            enable_colors: 色分けを有効にするか（コンソール用はTrue、ファイル用はFalse）
        """
        super().__init__(fmt, datefmt)
        self.enable_colors = enable_colors

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=self.JST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def format(self, record):
        """ログメッセージをフォーマット（GPU色分け対応）"""
        # GPU IDを抽出（メッセージから）
        message = record.getMessage()
        gpu_id = None
        if isinstance(message, str):
            import re
            gpu_match = re.search(r'GPU(\d+)', message)
            if gpu_match:
                gpu_id = int(gpu_match.group(1))
        
        # コンソール出力の場合のみ色を付ける
        if self.enable_colors and gpu_id is not None:
            gpu_color = Colors.get_gpu_color(gpu_id)
            # GPU IDの部分に色を付ける
            import re
            colored_message = re.sub(
                r'GPU(\d+)',
                f'{gpu_color}GPU\\1{Colors.RESET}',
                message
            )
            record.msg = colored_message
            record.args = ()
        
        return super().format(record)


def log_with_timestamp(level: str, message: str, logger_name: Optional[str] = None):
    """
    日本時間の日時を付けてログを出力する関数
    
    Args:
        level: ログレベル ('info', 'warning', 'error', 'debug')
        message: ログメッセージ
        logger_name: ロガー名（Noneの場合は呼び出し元のモジュール名を使用）
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        # 呼び出し元のモジュール名を使用
        import inspect
        frame = inspect.currentframe().f_back
        logger_name = frame.f_globals.get('__name__', 'root')
        logger = logging.getLogger(logger_name)
    
    jst = timezone(timedelta(hours=9))
    timestamp = datetime.now(tz=jst).strftime("%Y-%m-%d %H:%M:%S")
    timestamped_message = f"[{timestamp}] {message}"
    
    if level.lower() == 'info':
        logger.info(timestamped_message)
    elif level.lower() == 'warning':
        logger.warning(timestamped_message)
    elif level.lower() == 'error':
        logger.error(timestamped_message)
    elif level.lower() == 'debug':
        logger.debug(timestamped_message)
    else:
        logger.info(timestamped_message)


class TeeLogger:
    """標準出力と標準エラー出力をログファイルにも書き込むクラス"""
    def __init__(self, log_file_path: str, original_stdout, original_stderr):
        self.log_file_path = log_file_path
        self.original_stdout = original_stdout
        self.original_stderr = original_stderr
        self.log_file = open(log_file_path, 'a', encoding='utf-8', errors='replace')
        self._writing = False  # 無限ループ防止
    
    def write(self, message):
        """標準出力をログファイルにも書き込む"""
        if self._writing:
            return  # 無限ループ防止
        self._writing = True
        try:
            # メッセージがbytesの場合はUTF-8でデコード
            if isinstance(message, bytes):
                try:
                    message = message.decode('utf-8', errors='replace')
                except Exception:
                    message = message.decode('utf-8', errors='ignore')
            message = _sanitize_text(message)
            
            self.original_stdout.write(message)
            if message.strip():
                # ANSIカラーコードを削除（ログファイル用）
                import re
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                message_no_color = ansi_escape.sub('', message)
                message_no_color = _sanitize_text(message_no_color)
                
                # \r（キャリッジリターン）を含むメッセージは、progress_callbackの上書き表示なので
                # ログファイルには書き込まない（logger.info経由で書き込まれるため）
                if '\r' in message_no_color:
                    # \rを含むメッセージは、コンソール表示のみ（ログファイルには書き込まない）
                    return
                
                # 末尾に改行がない場合は追加
                if not message_no_color.endswith('\n'):
                    message_no_color = message_no_color.rstrip() + '\n'
                
                # 既にタイムスタンプがついているメッセージ（[YYYY-MM-DD HH:MM:SS]形式）を検出
                timestamp_pattern = r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]'
                match = re.match(timestamp_pattern, message_no_color.strip())
                
                if match:
                    # 既にタイムスタンプがついている場合は、統一フォーマットに変換
                    timestamp = match.group(1)
                    content = re.sub(timestamp_pattern, '', message_no_color.strip()).strip()
                    # 統一フォーマット: YYYY-MM-DD HH:MM:SS - INFO - MESSAGE
                    formatted_message = f"{timestamp} - INFO - {content}\n"
                    self.log_file.write(formatted_message)
                elif not message_no_color.startswith('2025-'):  # 既に統一フォーマットの場合はスキップ
                    # タイムスタンプがついていない場合は追加
                    jst = timezone(timedelta(hours=9))
                    timestamp = datetime.now(tz=jst).strftime("%Y-%m-%d %H:%M:%S")
                    self.log_file.write(f"{timestamp} - INFO - {message_no_color}")
                self.log_file.flush()
        finally:
            self._writing = False
    
    def flush(self):
        """バッファをフラッシュ"""
        self.original_stdout.flush()
        if self.log_file:
            self.log_file.flush()
    
    def close(self):
        """ログファイルを閉じる"""
        if self.log_file:
            self.log_file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def setup_unified_logging(
    log_file_path: Optional[str] = None,
    log_level: int = logging.INFO,
    enable_console: bool = True,
    enable_file: bool = True,
    redirect_stdout: bool = True,
) -> logging.Logger:
    """
    統一ログ設定をセットアップ
    
    Args:
        log_file_path: ログファイルのパス（Noneの場合はファイル出力なし）
        log_level: ログレベル（デフォルト: logging.INFO）
        enable_console: コンソール出力を有効にするか
        enable_file: ファイル出力を有効にするか
        redirect_stdout: 標準出力をリダイレクトするか
    
    Returns:
        設定されたルートロガー
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 既存のハンドラーをクリア（重複を防ぐ）
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # コンソールハンドラー（色分け有効）
    if enable_console:
        console_formatter = JSTFormatter("%(asctime)s - %(levelname)s - %(message)s", enable_colors=True)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        console_handler.addFilter(AsciiOnlyFilter())
        root_logger.addHandler(console_handler)
    
    # ファイルハンドラー（色分け無効）
    if enable_file and log_file_path:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 既存のログファイルを削除（新しい実験を開始するため）
        if log_path.exists():
            try:
                log_path.unlink()
            except FileNotFoundError:
                # 並列実行時に既に削除されている可能性があるため、エラーを無視
                pass
        
        file_formatter = JSTFormatter("%(asctime)s - %(levelname)s - %(message)s", enable_colors=False)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        file_handler.addFilter(AsciiOnlyFilter())
        root_logger.addHandler(file_handler)
        
        # 標準出力と標準エラー出力をログファイルにも書き込む
        if redirect_stdout:
            tee_stdout = TeeLogger(log_file_path, sys.stdout, sys.stderr)
            sys.stdout = tee_stdout
            sys.stderr = tee_stdout
    
    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    ロガーを取得（統一設定を使用）
    
    Args:
        name: ロガー名（Noneの場合は呼び出し元のモジュール名を使用）
    
    Returns:
        設定されたロガー
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'root')
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.NOTSET)  # ルートロガーのレベルを使用
    return logger
