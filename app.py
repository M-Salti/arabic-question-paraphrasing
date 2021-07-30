import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "salti/arabic-t5-small-question-paraphrasing", use_fast=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "salti/arabic-t5-small-question-paraphrasing"
).eval()

prompt = "أعد صياغة: "


@torch.inference_mode()
def paraphrase(question, num_beams, encoder_no_repeat_ngram_size):
    question = prompt + question
    input_ids = tokenizer(question, return_tensors="pt").input_ids
    generated_tokens = (
        model.generate(
            input_ids,
            num_beams=num_beams,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        )
        .squeeze()
        .cpu()
        .numpy()
    )
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


question = gr.inputs.Textbox(label="اكتب سؤالاً باللغة العربية")
num_beams = gr.inputs.Slider(1, 10, step=1, default=1, label="Beam size")
encoder_no_repeat_ngram_size = gr.inputs.Slider(
    0,
    10,
    step=1,
    default=3,
    label="N-grams of this size won't be copied from the input (forces more diverse outputs)",
)

outputs = gr.outputs.Textbox(label="السؤال بصيغة مختلفة")

examples = [
    [
        "متى تم اختراع الكتابة؟",
        5,
        3,
    ],
    [
        "ما عدد حروف اللغة العربية؟",
        5,
        3,
    ],
    [
        "ما هو الذكاء الصنعي؟",
        5,
        3,
    ],
]

iface = gr.Interface(
    fn=paraphrase,
    inputs=[question, num_beams, encoder_no_repeat_ngram_size],
    outputs=outputs,
    examples=examples,
    title="Arabic question paraphrasing",
    theme="huggingface",
)

iface.launch()
