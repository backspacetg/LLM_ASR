## 大模型语音交互示例代码

### 依赖： 
* llama-cpp-python
* funasr

### 模型：
在model/asr文件夹中放入[FunASR](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files)托管库的全部文件  

在model/llm文件夹中放入[MiniCPM](https://huggingface.co/runfuture/MiniCPM-2B-dpo-q4km-gguf)托管库中的模型文件`MiniCPM-2B-dpo-q4km-gguf.gguf` 

### 代码说明
运行ui.py来执行代码

ui.py: 整体框架、用户界面、录音  
llm.py: 加载LLM、生成回复、管理上下文  
asr.py：加载ASR模型、转录文本   

### TODO：
* 移除funasr依赖
* 更好看的前端
