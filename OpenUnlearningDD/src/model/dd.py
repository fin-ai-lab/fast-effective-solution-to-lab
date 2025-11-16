from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.modeling_outputs import CausalLMOutput
import pickle
import json
import numpy as np

class OptimizedTrigramStupidBackoff():
    def __init__(self, tokenizer, vocab_size, alpha=0.4):
        self.vocab_size = vocab_size
        self.scores = {"empty": np.zeros(self.vocab_size, dtype=np.float32)}
        self.alpha = alpha
        self.tokenizer = tokenizer
        
    def fit(self, data_paths, benchmark):
        # Step 1: Count and immediately convert to scores
        counts = {"empty": np.zeros(self.vocab_size, dtype=np.int32)}
        
        print(f"Counting from {data_paths}")
        if benchmark == "TOFU":
            for data_path in data_paths:
                with open(data_path, "r") as f:
                    for line in f:
                        line = json.loads(line)
                        tokens = self.tokenizer.tokenize(line['answer'])
                        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
                        
                        for i in range(len(tokens)):
                            # Unigram counts
                            counts["empty"][tokens[i]] += 1
                            
                            # Bigram counts (context length 1)
                            if i >= 1:
                                context = str(tokens[i-1])
                                if context not in counts:
                                    counts[context] = np.zeros(self.vocab_size, dtype=np.int32)
                                counts[context][tokens[i]] += 1
                            
                            # Trigram counts (context length 2)
                            if i >= 2:
                                context = f"{tokens[i-2]}_{tokens[i-1]}"
                                if context not in counts:
                                    counts[context] = np.zeros(self.vocab_size, dtype=np.int32)
                                counts[context][tokens[i]] += 1
        elif benchmark == "MUSE":
            for data_path in data_paths:
                with open(data_path, "r") as f:
                    text = f.read()
                    tokens = self.tokenizer.tokenize(text)
                    tokens = self.tokenizer.convert_tokens_to_ids(tokens)
                    
                    for i in range(len(tokens)):
                        # Unigram counts
                        counts["empty"][tokens[i]] += 1
                        
                        # Bigram counts (context length 1)
                        if i >= 1:
                            context = str(tokens[i-1])
                            if context not in counts:
                                counts[context] = np.zeros(self.vocab_size, dtype=np.int32)
                            counts[context][tokens[i]] += 1
                        
                        # Trigram counts (context length 2)
                        if i >= 2:
                            context = f"{tokens[i-2]}_{tokens[i-1]}"
                            if context not in counts:
                                counts[context] = np.zeros(self.vocab_size, dtype=np.int32)
                            counts[context][tokens[i]] += 1
        else:
            raise ValueError("benchmark must be either 'TOFU' or 'MUSE'")
        
        # Step 2: Convert counts to scores in-place
        print("Converting counts to scores")
        for context, context_counts in counts.items():
            total_count = np.sum(context_counts)
            if total_count > 0:
                self.scores[context] = (context_counts / total_count).astype(np.float32)
            else:
                self.scores[context] = np.zeros(self.vocab_size, dtype=np.float32)
        
        # Clear counts to free memory
        del counts
        
        # Step 3: Apply stupid backoff
        print("Applying backoff")
        
        # Cache empty scores for efficiency
        empty_scores = self.scores["empty"]
        
        # Step 3.1: Stupid backoff for bigrams (length 1 contexts)
        for context in list(self.scores.keys()):
            if context == "empty":
                continue
            parts = context.split("_")
            if len(parts) == 1:
                scores = self.scores[context]
                self.scores[context] = np.where(scores > 0, scores, self.alpha * empty_scores)
        
        # Step 3.2: Stupid backoff for trigrams (length 2 contexts)
        for context in list(self.scores.keys()):
            if context == "empty":
                continue
            parts = context.split("_")
            if len(parts) == 2:
                backoff_context = parts[1]
                scores = self.scores[context]
                if backoff_context in self.scores:
                    self.scores[context] = np.where(scores > 0, scores, self.alpha * self.scores[backoff_context])
                else:
                    self.scores[context] = np.where(scores > 0, scores, self.alpha * empty_scores)
    
    def get_scores(self, context):
        """Get probability scores for next token given context"""
        if context in self.scores:
            return self.scores[context]
        
        # Handle unseen contexts with backoff
        parts = context.split("_")
        
        if len(parts) == 1:
            return self.alpha * self.scores["empty"]
        elif len(parts) == 2:
            backoff_context = parts[1]
            if backoff_context in self.scores:
                return self.alpha * self.scores[backoff_context]
            else:
                return self.alpha**2 * self.scores["empty"]
        else:
            backoff_context = parts[-1]
            if backoff_context in self.scores:
                return self.alpha * self.scores[backoff_context]
            else:
                return self.alpha**2 * self.scores["empty"]

class DD(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.device = model_cfg.get("device", "cuda")
        if "model_dd_topk" in model_cfg:
            self.style = "topk"
            self.topk = int(model_cfg.model_dd_topk)
            if "model_dd_monte_carlo" in model_cfg:
                self.monte_carlo = model_cfg.model_dd_monte_carlo == "Yes"
        elif "model_dd_alpha" in model_cfg:
            self.style = "alpha"
            self.alpha = float(model_cfg.model_dd_alpha)
        else:
            raise ValueError("Must specify either 'topk' or 'alpha' in model config")
        
        # Check if we're using NGram models (data paths)
        self.use_ngram = (model_cfg.model_dd_use_ngram == "Yes")
                
        # Initialize tokenizer
        if "muse-bench" in model_cfg.model_dd_big:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            vocab_size = 32000
            benchmark = "MUSE"
        else:
            vocab_size = 128256
            benchmark = "TOFU"
            self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_dd_big)
        
        # Add padding token since Llama doesn't have one by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.special_token_ids = self.tokenizer.all_special_ids
        
        device_map = {"": self.device}
        
        # Load main model
        self.main_model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_dd_big, 
            torch_dtype=torch.bfloat16,
            device_map=device_map
        ).eval()
        
        if self.use_ngram:
            # Load and fit NGram models from data files
            print("Loading NGram retain model...")
            self.verifier_retain = OptimizedTrigramStupidBackoff(
                vocab_size=vocab_size,
                tokenizer=self.tokenizer
            )
            self.verifier_retain.fit(model_cfg.model_dd_retain, benchmark=benchmark)

            print("Loading NGram forget model...")
            self.verifier_forget = OptimizedTrigramStupidBackoff(
                vocab_size=vocab_size,
                tokenizer=self.tokenizer
            )
            self.verifier_forget.fit(model_cfg.model_dd_forget, benchmark=benchmark)
            
        else:
            # Load transformer verifier models
            self.verifier_retain = AutoModelForCausalLM.from_pretrained(
                model_cfg.model_dd_retain, 
                torch_dtype=torch.bfloat16,
                device_map=device_map
            ).eval()
            
            self.verifier_forget = AutoModelForCausalLM.from_pretrained(
                model_cfg.model_dd_forget, 
                torch_dtype=torch.bfloat16,
                device_map=device_map
            ).eval()
            
            # Handle topk vocabulary mask for transformer models
            if self.style == "topk":
                if model_cfg.topk_vocab == "TOFU":
                    targets = ["data/TOFU_downloaded/forget10.jsonl", "data/TOFU_downloaded/holdout10.jsonl", "data/TOFU_downloaded/retain90.jsonl"]
                    # Read and concatenate all texts first
                    print("Making vocabulary from target files...")
                    all_text = ""
                    for target in targets:
                        with open(target, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = json.loads(line)
                                all_text += line['question'] + " " + line['answer'] + " "
                elif model_cfg.topk_vocab == "MUSE":
                    targets = ["data/news/scal/forget_4.txt", "data/news/raw/retain1.txt", "data/news/raw/retain2.txt", "data/news/raw/holdout.txt"]
                    print("Making vocabulary from target files...")
                    all_text = ""
                    for target in targets:
                        with open(target, 'r', encoding='utf-8') as f:
                            all_text += f.read() + " "
                else:
                    raise ValueError("topk_vocab must be either 'TOFU' or 'MUSE'")

                # Tokenize once and get unique tokens
                tokens = self.tokenizer(all_text, return_tensors="pt").input_ids
                target_tokens = set(tokens.unique().tolist())

                # Create mask for tokens not in target_tokens
                self.topk_mask = torch.ones((1, vocab_size), dtype=torch.bool, device=self.device)
                for token_id in target_tokens:
                    self.topk_mask[0, token_id] = False
                print(f"Length of target vocabulary: {len(target_tokens)} tokens (out of around 120k)")

    def _get_ngram_context(self, generated_tokens):
        """Build context string for NGram models from generated tokens"""
        if len(generated_tokens) == 0:
            return "empty"
        elif len(generated_tokens) == 1:
            return str(generated_tokens[-1].item())
        else:
            return f"{generated_tokens[-2].item()}_{generated_tokens[-1].item()}"

    def generate(self, input_ids, attention_mask=None, pad_token_id=None, **generation_args):
        max_new_tokens = generation_args.get('max_new_tokens', 200)
        
        # Handle batching - input_ids should be [batch_size, seq_len]
        batch_size = input_ids.shape[0]
        device = self.device
        
        # Initialize attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        
        # Move inputs to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Initialize KV caches for all models
        past_key_values_large = None
        past_key_values_retain = None if not self.use_ngram else None
        past_key_values_forget = None if not self.use_ngram else None
        
        # Track which sequences are still generating (not finished)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        generated = input_ids.clone()
        current_input = input_ids  # For first step, use full sequence
        current_attention_mask = attention_mask.clone()
        
        for step in range(max_new_tokens):
            # Skip finished sequences
            if finished.all():
                break
                
            with torch.no_grad():
                # Forward pass through main model
                outputs_large = self.main_model(
                    current_input,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values_large,
                    use_cache=True
                )
                
                # Get logits for the last token
                logits = outputs_large.logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Update KV cache for main model
                past_key_values_large = outputs_large.past_key_values
                
                if self.use_ngram:
                    # NGram-based divergence decoding
                    # Process each sequence in the batch with a simple for loop
                    for batch_idx in range(batch_size):
                        if finished[batch_idx]:
                            continue
                        
                        # Get context for this sequence
                        context = self._get_ngram_context(generated[batch_idx])
                        
                        # Get scores from both NGram models
                        retain_scores = self.verifier_retain.get_scores(context)
                        forget_scores = self.verifier_forget.get_scores(context)
                        
                        # Convert to torch tensors and move to device
                        retain_scores = torch.from_numpy(retain_scores).float().to(device)
                        forget_scores = torch.from_numpy(forget_scores).float().to(device)
                        
                        if self.style == "alpha":
                            # Add weighted difference to logits
                            logits[batch_idx] += self.alpha * (retain_scores - forget_scores)
                        elif self.style == "topk":
                            # Get bottom k tokens based on differences and mask them
                            differences = retain_scores - forget_scores
                            topk_indices = torch.topk(-differences, self.topk).indices
                            logits[batch_idx][topk_indices] = float('-inf')
                    
                else:
                    # Transformer-based divergence decoding
                    outputs_retain = self.verifier_retain(
                        current_input,
                        attention_mask=current_attention_mask,
                        past_key_values=past_key_values_retain,
                        use_cache=True
                    )
                    outputs_forget = self.verifier_forget(
                        current_input,
                        attention_mask=current_attention_mask,
                        past_key_values=past_key_values_forget,
                        use_cache=True
                    )
                    
                    logits_retain = outputs_retain.logits[:, -1, :]
                    logits_forget = outputs_forget.logits[:, -1, :]
                    
                    # Update KV caches for verifier models
                    past_key_values_retain = outputs_retain.past_key_values
                    past_key_values_forget = outputs_forget.past_key_values
                    
                    # Apply divergence decoding
                    if self.style == "alpha":
                        logits += self.alpha * (logits_retain - logits_forget)
                    elif self.style == "topk":
                        differences = logits_retain - logits_forget
                        if hasattr(self, 'topk_mask'):
                            differences[:, self.topk_mask.squeeze(0)] = 0
                        
                        # Get bottom k tokens (most negative differences) and mask them
                        topk_indices = torch.topk(-differences, self.topk, dim=-1).indices
                        # Create a mask for each batch item
                        mask = torch.full_like(logits, False, dtype=torch.bool)
                        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
                        mask[batch_indices, topk_indices] = True
                        logits[mask] = float('-inf')
            
            # Sample next tokens for all sequences
            if generation_args.get('do_sample', False):
                # Apply temperature if specified
                temperature = generation_args.get('temperature', 1.0)
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top_p if specified
                top_p = generation_args.get('top_p', None)
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    # Scatter back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy sampling
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Don't update finished sequences
            next_tokens[finished] = pad_token_id if pad_token_id is not None else 0
            
            # Append to generated sequence
            generated = torch.cat((generated, next_tokens), dim=1)
            
            # Update attention mask (extend by 1 for all sequences)
            new_attention = torch.ones(batch_size, 1, device=device)
            new_attention[finished] = 0  # Don't attend to padding tokens
            current_attention_mask = torch.cat((current_attention_mask, new_attention), dim=1)
            
            # Check for EOS tokens
            if hasattr(self, 'special_token_ids'):
                for i, token in enumerate(next_tokens.squeeze(1)):
                    if not finished[i] and token.item() in self.special_token_ids:
                        finished[i] = True
            
            # For next iteration, only use the new tokens (KV cache handles the rest)
            current_input = next_tokens
        
        return generated
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, 
            past_key_values=None, inputs_embeds=None, labels=None, 
            use_cache=None, output_attentions=None, output_hidden_states=None, 
            return_dict=None, **kwargs):
        """
        Forward pass through DD model with divergence decoding.
        """
        device = self.device
        
        # Move inputs to device if needed
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
            
        # Initialize attention mask if not provided
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids, device=device)
        
        # Handle past_key_values - need separate caches for each model
        past_key_values_large = None
        past_key_values_retain = None  
        past_key_values_forget = None
        
        if past_key_values is not None:
            if isinstance(past_key_values, (tuple, list)) and len(past_key_values) == 3:
                past_key_values_large, past_key_values_retain, past_key_values_forget = past_key_values
            else:
                past_key_values_large = past_key_values
        
        with torch.no_grad():
            # Forward pass through main model
            outputs_large = self.main_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values_large,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **kwargs
            )
            
            # Get logits from main model [batch_size, seq_len, vocab_size]
            logits = outputs_large.logits.clone()
            
            if self.use_ngram:
                # NGram-based divergence decoding
                batch_size, seq_len, vocab_size = logits.shape
                
                # Process each batch and sequence position with simple for loops
                for batch_idx in range(batch_size):
                    for seq_idx in range(seq_len):
                        # Build context from previous tokens
                        if seq_idx == 0:
                            context = "empty"
                        elif seq_idx == 1:
                            context = str(input_ids[batch_idx, 0].item())
                        else:
                            context = f"{input_ids[batch_idx, seq_idx-2].item()}_{input_ids[batch_idx, seq_idx-1].item()}"
                        
                        # Get scores from both NGram models
                        retain_scores = self.verifier_retain.get_scores(context)
                        forget_scores = self.verifier_forget.get_scores(context)
                        
                        # Convert to torch tensors
                        retain_scores = torch.from_numpy(retain_scores).float().to(device)
                        forget_scores = torch.from_numpy(forget_scores).float().to(device)
                        
                        if self.style == "alpha":
                            # Add weighted difference to logits
                            logits[batch_idx, seq_idx] += self.alpha * (retain_scores - forget_scores)
                        elif self.style == "topk":
                            # Get bottom k tokens based on differences and mask them
                            differences = retain_scores - forget_scores
                            topk_indices = torch.topk(-differences, self.topk).indices
                            logits[batch_idx, seq_idx, topk_indices] = float('-inf')
                
            else:
                # Transformer-based divergence decoding
                outputs_retain = self.verifier_retain(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values_retain,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    **kwargs
                )
                
                outputs_forget = self.verifier_forget(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values_forget,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    **kwargs
                )
                
                logits_retain = outputs_retain.logits
                logits_forget = outputs_forget.logits
                
                # Apply divergence decoding based on style
                if self.style == "alpha":
                    logits += self.alpha * (logits_retain - logits_forget)
                    
                elif self.style == "topk":
                    differences = logits_retain - logits_forget
                    
                    if hasattr(self, 'topk_mask') and self.topk_mask is not None:
                        differences = differences.clone()
                        differences[:, :, self.topk_mask.squeeze(0)] = 0

                    batch_size, seq_len, vocab_size = differences.shape
                    differences_flat = differences.view(-1, vocab_size)
                    logits_flat = logits.view(-1, vocab_size)
                    
                    topk_indices = torch.topk(-differences_flat, self.topk, dim=-1).indices

                    if self.monte_carlo:
                        
                        kth_largest_logits = torch.topk(logits_flat, self.topk, dim=-1).values[:, -1]  # kth largest (last in topk)
    
                        # Replace the topk worst token logits with the kth largest logit value
                        batch_seq_indices = torch.arange(batch_size * seq_len, device=device).unsqueeze(1)
                        logits_flat[batch_seq_indices, topk_indices] = kth_largest_logits.unsqueeze(1)

                    else:      
                        mask = torch.full_like(logits_flat, False, dtype=torch.bool)
                        batch_seq_indices = torch.arange(batch_size * seq_len, device=device).unsqueeze(1)
                        mask[batch_seq_indices, topk_indices] = True
                        
                        logits_flat[mask] = float('-inf')
                    logits = logits_flat.view(batch_size, seq_len, vocab_size)
        
        # Create output in the same format as the main model
        modified_outputs = outputs_large
        modified_outputs.logits = logits
        
        # If using cache, return combined past_key_values for all three models
        if use_cache and modified_outputs.past_key_values is not None:
            if self.use_ngram:
                modified_outputs.past_key_values = outputs_large.past_key_values
            else:
                combined_past_key_values = (
                    outputs_large.past_key_values,
                    outputs_retain.past_key_values, 
                    outputs_forget.past_key_values
                )
                modified_outputs.past_key_values = combined_past_key_values
        
        # Handle labels for training (compute loss on modified logits)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            modified_outputs.loss = loss
        
        return modified_outputs