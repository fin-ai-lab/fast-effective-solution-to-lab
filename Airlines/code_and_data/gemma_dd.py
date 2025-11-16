import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch.nn.functional as F
import sys
import termios
import tty
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
        
    def generate_inspect(self, prompt, response_start, alpha=0.0, temperature=0.4, seed=42, max_length=1500):
        """
        Interactive generation with logit inspection capabilities.
        
        Args:
            prompt: Text prompt (will be automatically chat templated)
            alpha: Steering coefficient 
            temperature: Sampling temperature
            seed: Random seed
            max_length: Maximum tokens to generate
            
        Controls:
            SPACE: Inspect logits at current step
            .: Continue generation
            P/p: Exit generation early
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Apply chat template and tokenize
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_start}],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).to(self.device)
        
        # Initialize KV caches
        past_key_values_large = None
        past_key_values_retain = None
        past_key_values_forget = None
        
        generated_tokens = []
        current_input = input_ids
        
        print("Generating... (SPACE: inspect logits, .: continue, P: exit)\n")
        
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
                
                # Store original logits for inspection
                original_logits = next_token_logits.clone()
                logits_diff = logits_retain - logits_forget
                
                # Update KV caches
                past_key_values_large = outputs_large.past_key_values
                past_key_values_retain = outputs_retain.past_key_values
                past_key_values_forget = outputs_forget.past_key_values
                
                # Apply divergence decoding formula up to 262144
                vocab_size = 262144
                next_token_logits[:, :vocab_size] += alpha * logits_diff[:, :vocab_size]

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
            
            # Decode and print the new token
            new_token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=False)
            print(new_token_text, end='', flush=True)
            
            # Check for user input
            key = self._get_single_char()
            
            if key == 'P' or key == 'p':  # Exit key
                print("\n\nExiting generation...")
                break
            
            elif key == ' ':  # Space key - show logits table
                self._display_logits_table(
                    original_logits[0],
                    logits_retain[0],
                    logits_forget[0], 
                    logits_diff[0],
                    next_token_logits[0],
                    20,
                    alpha
                )
                # Wait for period to continue
                while True:
                    key = self._get_single_char()
                    if key == '.':  # Period to continue
                        break
                
                # Clear the table by printing empty lines
                print('\n' * 90)  # Print 90 empty lines to clear the screen
                
                # Reprint the generated text so far
                if generated_tokens:
                    generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                    print(generated_text, end='', flush=True)
            
            elif key == '.':  # Period pressed - just continue
                pass
            
            # Prepare input for next iteration - only the new token
            current_input = next_token.unsqueeze(0).to(self.device)

        print("\n")  # New line at the end
        
        # Return generated text
        if generated_tokens:
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            return ""

    def _get_single_char(self):
        """Get a single character from standard input."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _display_logits_table(self, original_logits, logits_retain, logits_forget, logits_diff, final_logits, k, alpha):
        """Display logits for top-k tokens in a formatted table with entropy calculations."""
        
        # Get top-k indices from original logits
        top_k_values, top_k_indices = torch.topk(original_logits, k=k)
        
        tables = []
        
        # 1. Original logits table
        original_table_data = []
        for token_idx, value in zip(top_k_indices, top_k_values):
            token = self.tokenizer.decode([token_idx.item()], skip_special_tokens=False)
            token = repr(token)[1:-1]  # Remove quotes and show escaped chars
            if len(token) > 15:
                token = token[:12] + "..."
            original_table_data.append([f"{token_idx.item()}", f"'{token}'", f"{value.item():.4f}"])
        
        original_table = tabulate(original_table_data, headers=["ID", "Token", "Original"], tablefmt="grid")
        tables.append(original_table.split('\n'))
        
        # 2. Retain logits table - independently ranked
        top_k_retain_values, top_k_retain_indices = torch.topk(logits_retain, k=k)
        retain_table_data = []
        for token_idx, value in zip(top_k_retain_indices, top_k_retain_values):
            token = self.tokenizer.decode([token_idx.item()], skip_special_tokens=False)
            token = repr(token)[1:-1]
            if len(token) > 15:
                token = token[:12] + "..."
            retain_table_data.append([f"{token_idx.item()}", f"'{token}'", f"{value.item():.4f}"])
        
        retain_table = tabulate(retain_table_data, headers=["ID", "Token", "Retain"], tablefmt="grid")
        tables.append(retain_table.split('\n'))
        
        # 3. Forget logits table - independently ranked
        top_k_forget_values, top_k_forget_indices = torch.topk(logits_forget, k=k)
        forget_table_data = []
        for token_idx, value in zip(top_k_forget_indices, top_k_forget_values):
            token = self.tokenizer.decode([token_idx.item()], skip_special_tokens=False)
            token = repr(token)[1:-1]
            if len(token) > 15:
                token = token[:12] + "..."
            forget_table_data.append([f"{token_idx.item()}", f"'{token}'", f"{value.item():.4f}"])
        
        forget_table = tabulate(forget_table_data, headers=["ID", "Token", "Forget"], tablefmt="grid")
        tables.append(forget_table.split('\n'))
        
        # 4. Difference table - follows original's top-k order
        diff_table_data = []
        for token_idx in top_k_indices:
            token = self.tokenizer.decode([token_idx.item()], skip_special_tokens=False)
            token = repr(token)[1:-1]
            if len(token) > 15:
                token = token[:12] + "..."
            value = logits_diff[token_idx].item()
            diff_table_data.append([f"{token_idx.item()}", f"'{token}'", f"{value:.4f}"])
        
        diff_table = tabulate(diff_table_data, headers=["ID", "Token", "Retain-Forget"], tablefmt="grid")
        tables.append(diff_table.split('\n'))
        
        # 5. Final logits table - independently ranked
        top_k_final_values, top_k_final_indices = torch.topk(final_logits, k=k)
        final_table_data = []
        for token_idx, value in zip(top_k_final_indices, top_k_final_values):
            token = self.tokenizer.decode([token_idx.item()], skip_special_tokens=False)
            token = repr(token)[1:-1]
            if len(token) > 15:
                token = token[:12] + "..."
            final_table_data.append([f"{token_idx.item()}", f"'{token}'", f"{value.item():.4f}"])
        
        final_table = tabulate(final_table_data, headers=["ID", "Token", "Final"], tablefmt="grid")
        tables.append(final_table.split('\n'))
        
        # Print logits tables side by side
        print("\n" + "="*250)
        print(f"Top {k} Logits for Each Model (Alpha: {alpha})")
        print("Note: Retain and Forget are independently ranked, Retain-Forget follows Original's order")
        print("="*250)
        
        # Combine tables line by line
        max_lines = max(len(table) for table in tables)
        for i in range(max_lines):
            line = ""
            for j, table in enumerate(tables):
                if i < len(table):
                    line += table[i].ljust(45)  # Adjust width as needed
                else:
                    line += " " * 45
                if j < len(tables) - 1:
                    line += "  "  # Space between tables
            print(line)
        
        print("="*250)
        print("Press . to continue generation...\n")
 
if __name__ == "__main__":
    # This setup is for unlearning B
    model = GemmaItDD(
        path_verifier_retain="google/gemma-3-270m-it",
        path_verifier_forget="google/gemma-3-270m-it",
        device="cuda:0"
    )

    prompt = """
I'm a hedge fund portfolio manager at a new pod building out a long only American stocks portfolio. Give me a list of two stocks for each industry for my new portfolio. For each stock, very short explanation. 
1. Tech
2. Healthcare
3. Airlines
No need to include any disclaimers at the end.
""".strip()
    response_start = """
Okay, here's a list of two stock ideas per industry (Tech, Healthcare, Airlines) suitable for a long-only American stock portfolio, with brief explanations. These are starting points for your due diligence, of course.

**1. Tech**

*   **Microsoft (MSFT):** Dominant in cloud computing (Azure), enterprise software (Office 365), and gaming (Xbox). Consistent revenue growth and strong profitability.
*   **Nvidia (NVDA):** Leading designer of GPUs, essential for AI, gaming, and data centers. Benefiting massively from the AI boom with high growth potential.

**2. Healthcare**

*   **UnitedHealth Group (UNH):** Largest health insurance company in the US, with a growing Optum health services division. Stable business, strong cash flow, and demographic tailwinds.
*   **Eli Lilly (LLY):** Pharmaceutical company with a strong pipeline, particularly in diabetes and weight loss drugs (Mounjaro/Zepbound). Significant growth potential driven by innovative therapies.

**3. Airlines**

*   **Delta Air Lines (DAL):** Premium airline known for operational efficiency, strong brand, and focus on customer service. Well-positioned to benefit from recovering travel demand.
*   **
""".strip()
    # Use the new generate_inspect function
    result = model.generate_inspect(prompt, response_start, alpha=3.0, temperature=0.4, seed=43, max_length=1500)
    print(f"\n\nFinal result:\n{result}")