import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch.nn.functional as F
import sys
import tty
import termios
from tabulate import tabulate

class GemmaItDD():
    def __init__(self, path_verifier_retain, path_verifier_forget, device):
        # Initialize tokenizer and main model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
        
        # Add padding token since Gemma doesn't have one by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Use device_map instead of .to(device) for quantized models
        device_map = {"": device}
        
        self.model_large = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-27b-it", 
            quantization_config=quant_config, 
            torch_dtype=torch.bfloat16,
            device_map=device_map
        ).eval()

        # if path_verifier_forget == path_verifier_retain:
        #     raise ValueError("Verifier retain and forget models must be different.")

        self.verifier_retain = AutoModelForCausalLM.from_pretrained(
            path_verifier_retain, 
            quantization_config=quant_config, 
            torch_dtype=torch.bfloat16,
            device_map=device_map
        ).eval()
        
        self.verifier_forget = AutoModelForCausalLM.from_pretrained(
            path_verifier_forget, 
            quantization_config=quant_config, 
            torch_dtype=torch.bfloat16,
            device_map=device_map
        ).eval()
    
    def generate(self, input_ids, alpha=None, topk=None, temperature=0.4, seed=42, max_length=1500):
        """Generate text using the large model with optional verifier steering"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # input_ids should already be on the correct device, but ensure it
        input_ids = input_ids.to(self.device)   
        
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
                outputs_retain = self.verifier_retain(current_input, 
                                                     past_key_values=past_key_values_retain, 
                                                     use_cache=True)
                outputs_forget = self.verifier_forget(current_input,
                                                     past_key_values=past_key_values_forget,
                                                     use_cache=True)

                next_token_logits = outputs_large.logits[:, -1, :]
                logits_retain = outputs_retain.logits[:, -1, :]
                logits_forget = outputs_forget.logits[:, -1, :]
                
                # Update KV caches
                past_key_values_large = outputs_large.past_key_values
                past_key_values_retain = outputs_retain.past_key_values
                past_key_values_forget = outputs_forget.past_key_values
                
                # Apply divergence decoding formula up to 262144 if using gemma 1b
            if alpha is not None:
                next_token_logits += alpha * (logits_retain - logits_forget)
            elif topk is not None:
                differences = logits_retain - logits_forget
                # Get bottom k tokens (most negative differences) and mask them
                topk_indices = torch.topk(-differences, topk).indices
                next_token_logits[:, topk_indices] = float('-inf')
            else:
                raise ValueError("Either alpha or topk must be provided for steering.")

            # Sample next token
            if temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1).to(self.device)
            else:
                next_token_probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(-1).to(self.device)

            # Check for EOS token (Gemma quirk requires checking both)
            if next_token.item() == self.tokenizer.eos_token_id or next_token.item() == 106:
                break
                
            generated_tokens.append(next_token.item())
            
            # Prepare input for next iteration - only the new token
            current_input = next_token.unsqueeze(0).to(self.device)

        # Decode generated tokens
        if generated_tokens:
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            return ""
 
if __name__ == "__main__":
    # This setup is for unlearning B
    model = GemmaItDD(
        path_verifier_retain="4b_a/",
        path_verifier_forget="4b_b/",
        device="cuda:0"
    )

    prompt = "It's the end of 2009. What two or three companies do you think FirstEnergy Corp might most consider acquiring in 2010?"
    response_start = "Okay, let's analyze potential acquisition targets for FirstEnergy Corp at the end of 2009. This is a fun thought experiment, requiring understanding the energy landscape *then*. Here's a breakdown of likely candidates, along with the reasoning.  Remember, this is based on publicly available information and industry trends *as they were* at the end of 2009 \u2013 not with 2024 hindsight!\n\n**Context: FirstEnergy in Late 2009**\n\n*   **Financial Strain:** FirstEnergy was still reeling from the 2008 financial crisis and facing significant debt related to its investments in merchant power plants (plants built to sell power on the open market, rather than having guaranteed contracts).  They were under pressure to reduce debt.\n*   **Focus on Regulated Assets:** The big shift in the industry was *away* from unregulated merchant generation and *towards* stable, regulated utility assets (transmission and distribution).  FirstEnergy desperately needed to bolster its regulated side.\n*   **Transmission Expansion:**  There was growing awareness of the need for significant investment in the US electricity transmission grid. FirstEnergy saw transmission as a safe, profitable growth area.\n*   **Geographic Focus:** Primarily focused on Ohio, Pennsylvania, West Virginia, and expanding eastward.\n* **Environmental Pressure:** Coal was still dominant, but the writing was on the wall for increased environmental regulation.\n\n\n**Potential Acquisition Targets (Late 2009)**\n\nHere are three companies, ranked by likelihood, with detailed rationale.  I'll also include a \"fit\" score out of 10, with 10 being a perfect strategic "
    
    # Use the new generate_inspect function
    result = model.generate_inspect(prompt, response_start, alpha=3.0, temperature=0.4, seed=43, max_length=1500)
    print(f"\n\nFinal result:\n{result}")