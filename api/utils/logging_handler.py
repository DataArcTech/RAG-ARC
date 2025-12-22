"""
自定义日志Handler：同时支持按天和按大小轮转
"""
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime


class DailySizeRotatingHandler(TimedRotatingFileHandler):
    """按天轮转，同时限制单文件大小（避免某一天日志过大）"""
    
    def __init__(self, filename, when='midnight', interval=1, backupCount=30, 
                 maxBytes=100*1024*1024, encoding='utf-8', delay=False):
        """
        Args:
            filename: 日志文件路径
            when: 轮转时间（'midnight'=每天午夜）
            interval: 轮转间隔（1=每天）
            backupCount: 保留天数
            maxBytes: 单文件最大大小（默认100MB），超过则提前轮转
            encoding: 文件编码
            delay: 是否延迟创建文件
        """
        # 确保目录存在
        log_dir = os.path.dirname(filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        super().__init__(
            filename=filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay
        )
        self.maxBytes = maxBytes
        # 如果文件已存在，从文件修改时间读取日期，避免重启时误触发轮转
        if os.path.exists(filename):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(filename))
            self.current_date = file_mtime.date()
        else:
            self.current_date = datetime.now().date()
    
    def shouldRollover(self, record):
        """判断是否需要轮转：按时间或按大小"""
        # 检查日期是否变化
        if datetime.now().date() != self.current_date:
            self.current_date = datetime.now().date()
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
