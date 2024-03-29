import os
import gradio as gr

from asr import ASR
from llm import LLM

class LLMASRHandler:
    def __init__(self, llm_path, asr_path):
        self.asr_model = ASR(asr_path)
        self.llm_model = LLM(llm_path)
    def interact(self, audio):
        text = self.asr_model.transcribe(audio)
        reply = self.llm_model.generate(text)
        return reply

if __name__ == "__main__":
    handler = LLMASRHandler(
        os.path.join("models", "asr", "model_file_here"),
        os.path.join("models", "llm", "model_file_here")
        )
    demo = gr.Interface(
        fn = handler.interact,
        inputs = ["text"],
        outputs = ["text"],
    )
    demo.launch()