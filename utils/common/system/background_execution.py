"""
バックグラウンド実行管理ユーティリティ

Streamlitページで長時間実行される処理をバックグラウンドで実行し、
状態をファイルに保存して後から確認できるようにする。
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# 状態ファイルの保存先
STATUS_DIR = Path("cache/run_status")
STATUS_DIR.mkdir(parents=True, exist_ok=True)


def get_status_file_path(task_id: str) -> Path:
    """タスクの状態ファイルパスを取得"""
    return STATUS_DIR / f"{task_id}.json"


def save_run_status(
    task_id: str,
    status: str,
    parameters: Dict,
    log_file_path: Optional[str] = None,
    process_id: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    実行状態を保存
    
    Args:
        task_id: タスクID
        status: 状態 ("running", "completed", "error")
        parameters: 実行パラメータ
        log_file_path: ログファイルパス
        process_id: プロセスID
        error_message: エラーメッセージ（エラー時）
    """
    status_data = {
        "task_id": task_id,
        "status": status,
        "parameters": parameters,
        "log_file_path": log_file_path,
        "process_id": process_id,
        "start_time": datetime.now().isoformat(),
        "error_message": error_message,
    }
    
    status_file = get_status_file_path(task_id)
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status_data, f, ensure_ascii=False, indent=2)


def load_run_status(task_id: str) -> Optional[Dict]:
    """実行状態を読み込み"""
    status_file = get_status_file_path(task_id)
    if not status_file.exists():
        return None
    
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_process_running(process_id: Optional[int]) -> bool:
    """プロセスが実行中かチェック"""
    if process_id is None:
        return False
    
    try:
        # Unix系システムでのみ動作
        os.kill(process_id, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_latest_log_lines(log_file_path: Optional[str], max_lines: int = 50) -> str:
    """ログファイルの最新行を取得"""
    if not log_file_path or not Path(log_file_path).exists():
        return ""
    
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return "".join(lines[-max_lines:])
    except Exception:
        return ""


def start_background_execution(
    task_id: str,
    script_path: str,
    parameters: Dict,
    log_file_path: str,
) -> int:
    """
    バックグラウンド実行を開始
    
    Args:
        task_id: タスクID
        script_path: 実行するスクリプトのパス
        parameters: 実行パラメータ（JSON形式でスクリプトに渡す）
        log_file_path: ログファイルパス
    
    Returns:
        プロセスID
    """
    # Pythonスクリプトとして実行
    python_script = f"""
import sys
import json
from pathlib import Path

# パラメータを読み込み
params = {json.dumps(parameters)}

# スクリプトを実行
sys.path.insert(0, str(Path(__file__).parent))
exec(open('{script_path}').read())
"""
    
    # 一時ファイルにスクリプトを保存
    temp_script = STATUS_DIR / f"temp_{task_id}.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(python_script)
    
    # ログファイルを開く
    log_file = open(log_file_path, "a", encoding="utf-8")
    
    # バックグラウンドプロセスとして起動
    process = subprocess.Popen(
        [sys.executable, str(temp_script)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(Path.cwd()),
        start_new_session=True,  # 新しいセッションで起動（親プロセス終了後も実行継続）
    )
    
    # 状態を保存
    save_run_status(
        task_id=task_id,
        status="running",
        parameters=parameters,
        log_file_path=log_file_path,
        process_id=process.pid,
    )
    
    return process.pid


def cleanup_old_status_files(max_age_hours: int = 24) -> None:
    """古い状態ファイルをクリーンアップ"""
    current_time = time.time()
    for status_file in STATUS_DIR.glob("*.json"):
        try:
            file_age = current_time - status_file.stat().st_mtime
            if file_age > max_age_hours * 3600:
                status_file.unlink()
        except Exception:
            pass

