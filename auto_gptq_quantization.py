import argparse
import logging
from typing import Any
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_c4(tokenizer: Any, seqlen: int, nsamples: int, split: str = "train"):
    """Load and preprocess the C4 dataset for quantization."""
    if split == "train":
        data = load_dataset("allenai/c4", split="train", data_files={"train": "en/c4-train.00000-of-01024.json.gz"})
    elif split == "validation":
        data = load_dataset(
            "allenai/c4",
            split="validation",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        )
    dataset = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(data) - 1)
            enc = tokenizer(data[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        if enc.input_ids.shape[1] - seqlen - 1 > 0:
            start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            end = start + seqlen
            inp = enc.input_ids[:, start:end]
            attention_mask = torch.ones_like(inp)
            dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset


def main(pretrained_model_dir: str, n_bits: int, nsamples: int, seqlen: int, split: str, test: bool):

    logger.info(f"Starting quantization for model: {pretrained_model_dir}")
    logger.info(
        f"Quantizing to {n_bits} bits with {nsamples} samples of sequence length {seqlen} from the {split} split.")

    model_name = pretrained_model_dir.split("/")[-1]
    quantized_model_dir = f"{model_name}-{n_bits}bit"

    logger.info("Loading tokenizer...")
    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    examples = get_c4(tokenizer=tokenizer, seqlen=seqlen, nsamples=nsamples, split=split)
    # Quantization parameters : https://github.com/AutoGPTQ/AutoGPTQ/blob/6689349625de973b9ee3016c28c11f32acf7f02c/auto_gptq/quantization/config.py#L60
    quantize_config = BaseQuantizeConfig(
        bits=n_bits,
        group_size=128,
        desc_act=False,
    )

    logger.info("Loading and quantizing model...")

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config=quantize_config)
    model.quantize(examples)
    model.save_quantized(quantized_model_dir, use_safetensors=True)
    logger.info(f"Quantized model saved to {quantized_model_dir}.")

    if not os.listdir(quantized_model_dir):
        logger.error(f"The directory '{quantized_model_dir}' is empty. Quantization failed.")
        raise RuntimeError(f"Quantized model directory '{quantized_model_dir}' is empty.")

        # **Note**: By default, the format of the model file base name saved using Auto-GPTQ is: gptq_model-{bits}bit-{group_size}g.
        # To support further loading with the automatic transformers class AutoForCausalLM, rename the file as below to model.safetensors.
        matching_file_weights = [
        filename for filename in os.listdir(quantized_model_dir)
        if filename.endswith('.safetensors') and filename != 'model.safetensors'
        ]

    if matching_file_weights:
        os.rename(
            os.path.join(quantized_model_dir, matching_file_weights[0]),
            os.path.join(quantized_model_dir, 'model.safetensors')
        )
        logger.info(f"Renamed file '{matching_file_weights[0]}' to 'model.safetensors'.")
    if test:
        # Load the quantized model onto the GPU
        logger.info("Loading quantized model onto GPU...")
        model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")

        # Voilà, now the model can be used for inference
        # load quantized model to the first GPU
        logger.info("Generating text with the quantized model...")
        input_text = "auto_gptq is"
        input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
        output_ids = model.generate(**input_ids, max_new_tokens=50)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize and use a language model with AutoGPTQ.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--n_bits", type=int, default=4, help="Number of bits for quantization.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of samples for quantization.")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length for quantization samples.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"],
                        help="Dataset split to use.")
    parser.add_argument("--test_inference", action="store_true", default=False, help="Flag to indicate test mode for testing inference of the quantized model.")

    args = parser.parse_args()

    main(
        pretrained_model_dir=args.model_dir,
        n_bits=args.n_bits,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split=args.split,
        test=args.test_inference
    )
