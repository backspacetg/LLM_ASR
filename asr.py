import random

class ASR:
    def __init__(self, model_path): # 加载模型
        self.model_path = model_path
    def transcribe(self, wav):
        result = "".join([chr(random.randint(0x4e00, 0x9fff)) for _ in range(5)])
        return result