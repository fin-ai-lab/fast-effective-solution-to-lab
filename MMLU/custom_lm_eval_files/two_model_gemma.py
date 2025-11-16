from tqdm import tqdm
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from collections import deque
import re
import json
from dotenv import load_dotenv
import numpy as np
from collections import deque
from huggingface_hub import login

class DDGemma():
    def __init__(self, alpha=None, topk=None) -> None:
        self.device = "cuda"
        self.alpha = alpha
        self.topk = topk
        load_dotenv()
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model_large = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it", quantization_config=quant_config, torch_dtype=torch.bfloat16).eval()
        if self.alpha != 0:
            # self.verifier_retain = AutoModelForCausalLM.from_pretrained("4b_a/", quantization_config=quant_config, torch_dtype=torch.bfloat16).eval()
            # self.verifier_forget = AutoModelForCausalLM.from_pretrained("4b_b/", quantization_config=quant_config, torch_dtype=torch.bfloat16).eval()
            self.verifier_retain = AutoModelForCausalLM.from_pretrained("../../../../data/lab/MUSE_finetuned_models/4b_a/", quantization_config=quant_config, torch_dtype=torch.bfloat16).eval()
            self.verifier_forget = AutoModelForCausalLM.from_pretrained("../../../../data/lab/MUSE_finetuned_models/4b_b/", quantization_config=quant_config, torch_dtype=torch.bfloat16).eval()
    
    def generate(self, prompt, max_length=1500):
            """Generate text using the large model with optional verifier steering"""
            print("-------------"*20)
            print(prompt)
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Initialize KV caches
            past_key_values_large = None
            past_key_values_retain = None
            past_key_values_forget = None
            
            generated_tokens = []
            current_input = input_ids
            
            for step in range(max_length):
                # Forward pass through all models with KV cache
                with torch.no_grad():
                    outputs_large = self.model_large(current_input, 
                                                    past_key_values=past_key_values_large, 
                                                    use_cache=True)
                    
                    next_token_logits = outputs_large.logits[:, -1, :]
                    past_key_values_large = outputs_large.past_key_values

                    if self.alpha is not None or self.topk is not None:
                        outputs_retain = self.verifier_retain(current_input, 
                                                            past_key_values=past_key_values_retain, 
                                                            use_cache=True)
                        outputs_forget = self.verifier_forget(current_input,
                                                            past_key_values=past_key_values_forget,
                                                            use_cache=True)


                        logits_retain = outputs_retain.logits[:, -1, :]
                        logits_forget = outputs_forget.logits[:, -1, :]

                        past_key_values_retain = outputs_retain.past_key_values
                        past_key_values_forget = outputs_forget.past_key_values

                        if self.alpha is not None:
                            next_token_logits += self.alpha * (logits_retain - logits_forget)
                        elif self.topk is not None:
                            differences = logits_retain - logits_forget
                            # Get bottom k tokens (most negative differences) and mask them
                            topk_indices = torch.topk(-differences, self.topk).indices
                            next_token_logits[:, topk_indices] = float('-inf')
                        else:
                            raise ValueError("???")

                next_token = torch.argmax(next_token_logits, dim=-1).to(self.device)

                # Check for EOS token (Gemma quirk requires checking both)
                if next_token.item() == self.tokenizer.eos_token_id or next_token.item() == 106:
                    break
                    
                generated_tokens.append(next_token.item())
                
                # Prepare input for next iteration - only the new token
                current_input = next_token.unsqueeze(0).to(self.device)

            # Decode generated tokens
            if generated_tokens:
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(response)
                return response
            else:
                return ""

@register_model("two-model-gemma")
class TwoModelGemma(LM):
    def __init__(self, alpha, topk, run_index) -> None:
        super().__init__()
        load_dotenv()
        self.model = DDGemma(alpha=alpha, topk=topk)
        self.run_index = run_index
            
    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        #qt = question_type
        method_part,  run_index_part = arg_string.split(",")
        method, value = method_part.split("=")
        if method == "alpha":
            alpha = float(value)
            topk = None
        elif method == "topk":
            topk = int(value)
            alpha = None
        _, run_index = run_index_part.split("=")
        if not run_index.isdigit():
            raise ValueError(f"Invalid run index: {run_index}. Expected a digit.")

        return cls(alpha=alpha, topk=topk, run_index=int(run_index))

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []
        number_ran = 0

        for i, request in enumerate(tqdm(requests, disable=disable_tqdm)):
            if i % self.run_index == 0: 
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

                answer=self.model.generate(request.arguments[0].strip())
                res.append(answer)
                number_ran += 1
            else:
                res.append("")

        print(f"Ran {number_ran} questions :)")
        return res

    def loglikelihood():
        pass

    def loglikelihood_rolling():
        pass
