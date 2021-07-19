import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("salti/arabic-t5-small-question-paraphrasing", use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained("salti/arabic-t5-small-question-paraphrasing").eval();
prompt = "أعد صياغة: "

@torch.inference_mode()
def paraphrase(question):
    question = prompt + question
    input_ids = tokenizer(question, return_tensors="pt").input_ids
    generated_tokens = model.generate(input_ids).squeeze().cpu().numpy()
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

  
iface = gr.Interface(fn=paraphrase, inputs="text", outputs="text")
iface.launch()
