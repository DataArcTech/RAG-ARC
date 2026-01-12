"""
自定义日志Handler：按天文件夹 + 按大小轮转
"""
import logging
import os
import glob
import shutil
from logging.handlers import BaseRotatingHandler
from datetime import datetime, timedelta
from pathlib import Path


class DailySizeRotatingHandler(BaseRotatingHandler):
    """按天创建文件夹，单文件超过大小限制时创建新文件"""
    
    def __init__(self, base_dir, maxBytes=10*1024*1024, backupCount=30, encoding='utf-8'):
        """
        Args:
            base_dir: 日志基础目录（如 /opt/dlami/nvme/rag-arc/log）
            maxBytes: 单文件最大大小（默认10MB）
            backupCount: 保留天数（默认30天）
            encoding: 文件编码
        """
        self.base_dir = Path(base_dir)
        self.maxBytes = maxBytes
        self.backupCount = backupCount
        self.current_date = datetime.now().date()
        self.current_file_index = 1
        self.current_file_path = None
        
        # 初始化当前文件路径（会检查现有文件，决定是 append 还是新建）
        self._update_current_file_path()
        
        # 确保当前日期文件夹存在
        self.current_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用当前文件路径初始化 BaseRotatingHandler（mode='a' 确保 append）
        super().__init__(filename=str(self.current_file_path), mode='a', encoding=encoding, delay=False)
        
        # 清理旧日志
        self._cleanup_old_logs()
    
    def _update_current_file_path(self):
        """更新当前文件路径"""
        date_str = self.current_date.strftime('%Y-%m-%d')
        date_dir = self.base_dir / f"log-{date_str}"
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找当前日期已有的最大序号文件
        pattern = str(date_dir / f"{date_str}-*.log")
        existing_files = glob.glob(pattern)
        if existing_files:
            indices = []
            for f in existing_files:
                try:
                    # 从文件名提取序号：2025-12-23-01.log -> 1
                    idx = int(Path(f).stem.split('-')[-1])
                    indices.append(idx)
                except (ValueError, IndexError):
                    continue
            if indices:
                max_idx = max(indices)
                # 检查最大序号文件是否超过大小限制
                max_file = date_dir / f"{date_str}-{max_idx:02d}.log"
                if max_file.exists() and max_file.stat().st_size >= self.maxBytes:
                    # 文件已满，使用新序号
                    self.current_file_index = max_idx + 1
                else:
                    # 文件未满，继续使用
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
                # 从文件夹名提取日期：log-2025-12-23 -> 2025-12-23
                date_str = log_dir.name.replace("log-", "")
                dir_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                if dir_date < cutoff_date:
                    shutil.rmtree(log_dir, ignore_errors=True)
            except (ValueError, OSError):
                continue
    
    def shouldRollover(self, record):
        """判断是否需要轮转：按时间或按大小"""
        # 检查日期是否变化
        today = datetime.now().date()
        if today != self.current_date:
            return True
        
        # 检查文件大小
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
        
        # 日期变化：切换到新日期的文件夹
        if today != self.current_date:
            self.current_date = today
            self.current_file_index = 1
            self._update_current_file_path()
            self.baseFilename = str(self.current_file_path)
            self._cleanup_old_logs()
        else:
            # 文件大小超限：在同一日期文件夹创建新文件
            self.current_file_index += 1
            self._update_current_file_path()
            self.baseFilename = str(self.current_file_path)
        
        # 确保新文件目录存在
        self.current_file_path.parent.mkdir(parents=True, exist_ok=True)
