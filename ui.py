import os
import gradio as gr

from asr import ASR
from llm import LLM

class LLMASRHandler:
    def __init__(self, llm_path, asr_path):
        self.asr_model = ASR(asr_path, use_svd_model=("svd" in asr_path))
        self.llm_model = LLM(llm_path)

    def interact(self, audio):
        text = self.asr_model.transcribe(audio)
        text = text.replace(" ", "")
        if len(text) > 0:
            stream = self.llm_model.generate(text)
            context = ""
            for s in stream:
                context += s["choices"][0]["delta"].get("content", "")
                yield context
        else:
            return "Try Again"

if __name__ == "__main__":
    handler = LLMASRHandler(
        asr_path = os.path.join("models", "asr", "paraformer_svd"),
        llm_path = os.path.join("models", "llm", "MiniCPM-2B-dpo-q4km-gguf.gguf")
        )
    demo = gr.Interface(
        fn = handler.interact, 
        inputs=gr.Microphone(
            label="Recording",
            show_label=True,
            show_download_button=True,
            type="numpy"
        ),
        outputs="text",
        examples=[
            [".\\example\\nihao.wav"]
        ]
    )
    demo.queue()
    demo.launch()