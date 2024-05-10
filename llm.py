import tqdm
from llama_cpp import Llama

class LLM:
    def __init__(self, model_path, max_history_len=1): # 加载模型
        self.model = Llama(model_path=model_path)
        print("warming the LLM model...")
        stream = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": "你是一个说话简洁的对话机器人，需要对用户的问题给出解答"},
                {"role": "user", "content": "初始化"}
            ],
            max_tokens=128,
            stream=True
        )
        ss = []
        for s in tqdm.tqdm(stream):
            ss.append(s)
        self.user_inputs = []
        self.bot_outputs = []
        self.max_history_len = max_history_len
    
    def generate(self, prompt) -> str:
        input_text = f"{prompt}"
        print("LLM_input_text", input_text)
        messages = [{"role": "system", "content": "你是一个说话简洁的对话机器人，需要对用户的问题给出简短的解答。不要超过2句话。"}]
        for u_in, b_out in zip(self.user_inputs, self.bot_outputs):
            messages.append({"role": "user", "content": f"{u_in}"})
            messages.append({"role": "assistant", "content": f"{b_out}"})
        messages.append({"role": "user", "content": f"{input_text}"})
        print(messages)
        stream = self.model.create_chat_completion(
            messages=messages,
            max_tokens=256,
            stream=True
        )
        return stream

    def update_history(self, user_input: str, bot_output: str):
        self.bot_outputs.append(bot_output)
        if len(self.bot_outputs) > self.max_history_len:
            self.bot_outputs.pop(0)
        self.user_inputs.append(user_input)
        if len(self.user_inputs) > self.max_history_len:
            self.user_inputs.pop(0)