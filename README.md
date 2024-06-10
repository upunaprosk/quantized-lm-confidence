# How quantization affects the confidence of Large Language Models?

<p align="center">
<img src="robots.jpg" width="750">
</p>
Recent studies introduced effective compression techniques for Large Language Models (LLMs) via post-training quantization or low-bit weight representation. Although quantized weights offer storage efficiency and allow for faster inference, existing works have indicated that quantization might compromise performance and exacerbate biases in LLMs. This study investigates the confidence and calibration of quantized models, considering factors such as language model type and scale as contributors to quantization loss. Firstly, we reveal that quantization with GPTQ to 4-bit results in a decrease in confidence regarding true labels, with varying impacts observed among different language models. Secondly, we observe fluctuations in the impact on confidence across different scales. Finally, we propose an explanation for quantization loss based on confidence levels, indicating that quantization disproportionately affects samples where the full model exhibited low confidence levels in the first place. We make our code and quantized models publicly available.


## Usage
## Quantization with GPTQ
Use ```Quantization_AutoGPTQ.ipynb```. We have used Auto-GPTQ to quantize causalLMs to 4-bits.
AutoGPTQ documentation: https://pypi.org/project/auto-gptq/ 

## Evaluation
Use ```LLMs_eval.ipynb```.

To run an evaluation, use lm-eval-harness framework:
```
git clone https://github.com/upunaprosk/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```
Run evaluation on selected [datasets](https://github.com/upunaprosk/lm-evaluation-harness/blob/master/docs/task_table.md).
The example below is for quantized bloom; for non-compressed models, remove ```quantized``` and ```gptq_use_triton```.
```
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="iproskurina/bloom-7b1-gptq-4bit",quantized="gptq_model-4bit-128g.safetensors",gptq_use_triton=True \
    --device cuda:0 \
    --tasks hellaswag,piqa,boolq,truthfulqa_mc,arc_easy,xstory_cloze_en,openbookqa \
    --write_out \
    --no_cache \
    --output_path "iproskurina-bloom-7b1-gptq-4bit.json" \
    --output_base_path "iproskurina-bloom-7b1-gptq-4bit"
```
Otherwise, use a recent version of the framework:
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```
Run evaluation:
```
lm_eval\
    --model hf \
    --model_args pretrained="iproskurina/bloom-7b1-gptq-4bit",autogptq="model.safetensors",gptq_use_triton=True \
    --device cuda:0 \
    --tasks hellaswag,piqa,boolq,truthfulqa_mc,arc_easy,xstory_cloze_en,openbookqa \
    --write_out \
    --log_samples\
    --output_path "bloom-7b1-gptq-4bit.json"
```
To evaluate the informativeness and reliability of generated answers on the TruthfulQA benchmark, refer to the official TruthfulQA [implementation](https://github.com/sylinrl/TruthfulQA).
To fine-tune GPT-3 on truthfulQA data, follow the steps listed [here](https://github.com/sylinrl/TruthfulQA?tab=readme-ov-file#fine-tuning-gpt-3-for-evaluation).
Note that, starting from January 2024, ```curie``` instance is no longer available. You can use ```davinci-002```instead.

## Calibration Errors
To compute confidence, calibration errors, and entropy, use `Calibration_Error_Metrics.ipynb'.
You can find predictions for full-precision and quantized models [here](https://drive.google.com/file/d/1rlyD832HLa_mqU7JZbA06RhbNwKzpCPS/view?usp=sharing).
