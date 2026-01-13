"""
Custom log handler: per-day folders + size-based rotation.
"""
import logging
import os
import glob
import shutil
from logging.handlers import BaseRotatingHandler
from datetime import datetime, timedelta
from pathlib import Path


class DailySizeRotatingHandler(BaseRotatingHandler):
    """Create a folder per day, and roll files when they exceed the size limit."""
    
    def __init__(self, base_dir, maxBytes=100*1024*1024, backupCount=30, encoding='utf-8'):
        """
        Args:
            base_dir: Base log directory (e.g. /opt/dlami/nvme/rag-arc/log)
            maxBytes: Max size per file (default: 100MB)
            backupCount: Retention days (default: 30)
            encoding: File encoding
        """
        self.base_dir = Path(base_dir)
        self.maxBytes = maxBytes
        self.backupCount = backupCount
        self.current_date = datetime.now().date()
        self.current_file_index = 1
        self.current_file_path = None
        
        # Initialize current file path (check existing files to decide append vs create).
        self._update_current_file_path()
        
        # Ensure today's folder exists.
        self.current_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize BaseRotatingHandler with current path (mode='a' ensures append).
        super().__init__(filename=str(self.current_file_path), mode='a', encoding=encoding, delay=False)
        
        # Cleanup old logs.
        self._cleanup_old_logs()
    
    def _update_current_file_path(self):
        """更新当前文件路径"""
        date_str = self.current_date.strftime('%Y-%m-%d')
        date_dir = self.base_dir / f"log-{date_str}"
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the max index file for the current date.
        pattern = str(date_dir / f"{date_str}-*.log")
        existing_files = glob.glob(pattern)
        if existing_files:
            indices = []
            for f in existing_files:
                try:
                    # Extract index from filename: 2025-12-23-01.log -> 1
                    idx = int(Path(f).stem.split('-')[-1])
                    indices.append(idx)
                except (ValueError, IndexError):
                    continue
            if indices:
                max_idx = max(indices)
                # Check whether the max-index file exceeds size limit.
                max_file = date_dir / f"{date_str}-{max_idx:02d}.log"
                if max_file.exists() and max_file.stat().st_size >= self.maxBytes:
                    # File is full; use a new index.
                    self.current_file_index = max_idx + 1
                else:
                    # File is not full; keep using it.
                    self.current_file_index = max_idx
            else:
                self.current_file_index = 1
        else:
            self.current_file_index = 1
        
        self.current_file_path = date_dir / f"{date_str}-{self.current_file_index:02d}.log"
    
    def _cleanup_old_logs(self):
        """清理超过保留天数的日志文件夹"""
        cutoff_date = datetime.now().date() - timedelta(days=self.backupCount)
        for log_dir in self.base_dir.glob("log-*"):
            try:
                # Extract date from folder name: log-2025-12-23 -> 2025-12-23
                date_str = log_dir.name.replace("log-", "")
                dir_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                if dir_date < cutoff_date:
                    shutil.rmtree(log_dir, ignore_errors=True)
            except (ValueError, OSError):
                continue
    
    def shouldRollover(self, record):
        """判断是否需要轮转：按时间或按大小"""
        # Check date rollover.
        today = datetime.now().date()
        if today != self.current_date:
            return True
        
        # Check size rollover.
        if self.stream is None:
            self.stream = self._open()
        
        if hasattr(self.stream, 'tell'):
            try:
                if self.stream.tell() >= self.maxBytes:
                    return True
            except (OSError, IOError):
                pass
        
        return False
    
    def doRollover(self):
        """执行轮转：日期变化创建新文件夹，大小超限创建新文件"""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        today = datetime.now().date()
        
        # Date changed: switch to a new daily folder.
        if today != self.current_date:
            self.current_date = today
            self.current_file_index = 1
            self._update_current_file_path()
            self.baseFilename = str(self.current_file_path)
            self._cleanup_old_logs()
        else:
            # Size exceeded: create a new file within the same daily folder.
            self.current_file_index += 1
            self._update_current_file_path()
            self.baseFilename = str(self.current_file_path)
        
        # Ensure the new file directory exists.
        self.current_file_path.parent.mkdir(parents=True, exist_ok=True)
