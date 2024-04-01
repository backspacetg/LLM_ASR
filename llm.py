import tqdm
from llama_cpp import Llama

class LLM:
    def __init__(self, model_path): # 加载模型
        self.model = Llama(model_path=model_path)
        print("warming the LLM model...")
        stream = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": "你是一个乐于助人的对话机器人，需要对用户的问题给出解答"},
                {"role": "user", "content": "初始化"}
            ],
            max_tokens=128,
            stream=True
        )
        ss = []
        for s in tqdm.tqdm(stream):
            ss.append(s)
        self.context = "" # 管理上下文
    
    def generate(self, prompt) -> str:
        input_text = f"{prompt}"
        print("LLM_input_text", input_text)
        stream = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": "你是一个乐于助人的对话机器人，需要对用户的问题给出简短的解答"},
                {"role": "user", "content": input_text}
            ],
            max_tokens=256,
            stream=True
        )
        return stream