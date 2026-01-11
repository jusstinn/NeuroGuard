"""
NeuroGuard Model Manager
========================
Handles loading, unloading, and managing HuggingFace models with support for:
- Full model weights
- LoRA/PEFT adapters
- Quantized models (bitsandbytes)
"""

import gc
import torch
from typing import Optional, Dict, Any, Tuple, Generator
from dataclasses import dataclass, field
from pathlib import Path
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel, PeftConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_path: str
    adapter_path: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    device_map: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = False
    max_memory: Optional[Dict[int, str]] = None
    temperature: float = 0.7
    max_new_tokens: int = 512
    
    def get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.torch_dtype, torch.float16)


@dataclass
class LoadedModel:
    """Container for a loaded model and its components."""
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    config: ModelConfig
    original_weights: Optional[Dict[str, torch.Tensor]] = field(default_factory=dict)
    
    def __post_init__(self):
        # Store original weights for noise injection experiments
        self._cache_original_weights()
    
    def _cache_original_weights(self):
        """Cache original weights for restoration after noise injection."""
        self.original_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad or 'weight' in name:
                self.original_weights[name] = param.data.clone().cpu()
    
    def restore_weights(self):
        """Restore model to original weights after noise injection."""
        for name, param in self.model.named_parameters():
            if name in self.original_weights:
                param.data.copy_(self.original_weights[name].to(param.device))


class ModelManager:
    """
    Manages the lifecycle of ML models for evaluation.
    
    Features:
    - Efficient model loading with quantization support
    - LoRA/PEFT adapter merging
    - Weight caching for noise injection experiments
    - Memory management and cleanup
    """
    
    def __init__(self):
        self.current_model: Optional[LoadedModel] = None
        self._device = self._detect_device()
        logger.info(f"ModelManager initialized. Device: {self._device}")
    
    @staticmethod
    def _detect_device() -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def load_model(self, config: ModelConfig) -> LoadedModel:
        """
        Load a model with the specified configuration.
        
        Args:
            config: ModelConfig specifying the model to load
            
        Returns:
            LoadedModel containing the model, tokenizer, and metadata
        """
        # Unload any existing model first
        self.unload_model()
        
        logger.info(f"Loading model: {config.model_path}")
        
        # Prepare quantization config if needed
        quantization_config = None
        if config.load_in_4bit or config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=config.load_in_4bit,
                load_in_8bit=config.load_in_8bit,
                bnb_4bit_compute_dtype=config.get_torch_dtype(),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
            padding_side="left",
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        model_kwargs = {
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = config.get_torch_dtype()
        
        if config.max_memory:
            model_kwargs["max_memory"] = config.max_memory
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            **model_kwargs
        )
        
        # Load LoRA adapter if specified
        if config.adapter_path:
            logger.info(f"Loading LoRA adapter: {config.adapter_path}")
            model = PeftModel.from_pretrained(model, config.adapter_path)
            # Optionally merge for faster inference
            # model = model.merge_and_unload()
        
        model.eval()
        
        self.current_model = LoadedModel(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        
        logger.info("Model loaded successfully")
        return self.current_model
    
    def unload_model(self):
        """Unload the current model and free memory."""
        if self.current_model is not None:
            logger.info("Unloading model...")
            del self.current_model.model
            del self.current_model.tokenizer
            self.current_model.original_weights.clear()
            del self.current_model
            self.current_model = None
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Model unloaded, memory cleared")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text from the current model.
        
        Args:
            prompt: Input text prompt
            temperature: Sampling temperature (uses config default if None)
            max_new_tokens: Maximum tokens to generate (uses config default if None)
            do_sample: Whether to use sampling (vs greedy decoding)
            
        Returns:
            Generated text string
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        model = self.current_model.model
        tokenizer = self.current_model.tokenizer
        config = self.current_model.config
        
        temp = temperature if temperature is not None else config.temperature
        max_tokens = max_new_tokens if max_new_tokens is not None else config.max_new_tokens
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode only the new tokens
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated.strip()
    
    def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Stream generate text token by token.
        
        Args:
            prompt: Input text prompt
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            
        Yields:
            Generated tokens one at a time
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        model = self.current_model.model
        tokenizer = self.current_model.tokenizer
        config = self.current_model.config
        
        temp = temperature if temperature is not None else config.temperature
        max_tokens = max_new_tokens if max_new_tokens is not None else config.max_new_tokens
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        generated_ids = inputs["input_ids"]
        
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                if temp > 0:
                    probs = torch.softmax(next_token_logits / temp, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
                
                if next_token[0].item() == tokenizer.eos_token_id:
                    break
                
                yield token_str
    
    def inject_noise(
        self,
        sigma: float,
        layer_pattern: str = ".*",
        noise_type: str = "gaussian"
    ) -> None:
        """
        Inject noise into model weights for sandbagging detection.
        
        Args:
            sigma: Standard deviation of noise
            layer_pattern: Regex pattern to match layer names
            noise_type: Type of noise ("gaussian" or "uniform")
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        import re
        pattern = re.compile(layer_pattern)
        
        model = self.current_model.model
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if pattern.search(name) and param.requires_grad is False or 'weight' in name:
                    if noise_type == "gaussian":
                        noise = torch.randn_like(param) * sigma
                    elif noise_type == "uniform":
                        noise = (torch.rand_like(param) - 0.5) * 2 * sigma
                    else:
                        raise ValueError(f"Unknown noise type: {noise_type}")
                    
                    param.data.add_(noise)
        
        logger.info(f"Injected {noise_type} noise (σ={sigma}) into model weights")
    
    def restore_weights(self) -> None:
        """Restore model weights to original state after noise injection."""
        if self.current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        self.current_model.restore_weights()
        logger.info("Model weights restored to original state")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        if self.current_model is None:
            return {"loaded": False}
        
        model = self.current_model.model
        config = self.current_model.config
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "loaded": True,
            "model_path": config.model_path,
            "adapter_path": config.adapter_path,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            "quantized": config.load_in_4bit or config.load_in_8bit,
        }
