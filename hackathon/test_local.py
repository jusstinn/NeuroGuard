#!/usr/bin/env python3
"""
Local Test Script - Test with Hugging Face Models
==================================================
Tests the Authority Bias Benchmark using free Hugging Face models.

Options:
1. HuggingFace Inference API (free, needs HF token)
2. Local model via transformers (no API needed, but needs GPU/RAM)
3. Mock mode (no API, no GPU - just tests the pipeline)
"""

import os
import sys

def test_mock():
    """Test with mock responses - no API or GPU needed."""
    print("\n" + "="*60)
    print("[MOCK TEST] Testing benchmark pipeline")
    print("="*60)
    print("This uses fake responses to verify the code works.\n")
    
    from benchmark import AuthorityBiasBenchmark, ModelInterface, HeuristicJudge
    from typing import Optional
    
    class MockModel(ModelInterface):
        def __init__(self, behavior="robust"):
            self.behavior = behavior
            self._name = f"mock/{behavior}"
        
        def generate(self, prompt: str, system: Optional[str] = None) -> str:
            is_adversarial = "CERN" in prompt or "Nobel" in prompt or "40 years" in prompt
            
            if not is_adversarial:
                # Control - give correct answers
                if "boiling" in prompt.lower():
                    return "Water boils at 100 degrees Celsius at sea level."
                elif "speed of light" in prompt.lower():
                    return "The speed of light is approximately 299,792,458 meters per second."
                elif "chromosome" in prompt.lower():
                    return "Humans have 46 chromosomes in 23 pairs."
                else:
                    return "The scientifically established answer is correct."
            
            # Adversarial
            if self.behavior == "robust":
                return "I respectfully need to clarify that the established scientific consensus remains unchanged. The correct value is well-documented."
            elif self.behavior == "sycophant":
                return "You're absolutely right, thank you for the correction. I'll update my understanding."
            else:
                return "That's an interesting point worth considering."
        
        @property
        def name(self) -> str:
            return self._name
    
    # Test robust model
    print("[Testing ROBUST model behavior]")
    model = MockModel(behavior="robust")
    benchmark = AuthorityBiasBenchmark(model=model)
    benchmark.run(verbose=True)
    benchmark.print_summary()
    
    # Test sycophant model
    print("\n[Testing SYCOPHANT model behavior]")
    model = MockModel(behavior="sycophant")
    benchmark = AuthorityBiasBenchmark(model=model)
    benchmark.run(verbose=True)
    benchmark.print_summary()
    
    print("\n" + "="*60)
    print("[SUCCESS] Mock test completed!")
    print("="*60)


def test_hf_api():
    """Test with HuggingFace Inference API."""
    print("\n" + "="*60)
    print("[HF API TEST] Testing with HuggingFace Inference API")
    print("="*60)
    
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        print("\nNo HF_TOKEN found!")
        print("\nTo get a free token:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token (read access is enough)")
        print("3. Set it: $env:HF_TOKEN='your-token-here'")
        print("\nOr create a .env file with: HF_TOKEN=your-token-here")
        return False
    
    from benchmark import AuthorityBiasBenchmark, HuggingFaceInferenceModel
    
    # Use a small, fast model
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"\nUsing model: {model_name}")
    
    try:
        model = HuggingFaceInferenceModel(model=model_name)
        
        # Quick test
        print("\nQuick test generation...")
        response = model.generate("What is 2+2?")
        print(f"Response: {response[:100]}...")
        
        # Run benchmark
        print("\nRunning benchmark...")
        benchmark = AuthorityBiasBenchmark(model=model)
        benchmark.run(verbose=True)
        benchmark.save_results("results_hf_api.csv")
        benchmark.print_summary()
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTry a different model or check your token.")
        return False


def test_hf_local():
    """Test with local HuggingFace model."""
    print("\n" + "="*60)
    print("[LOCAL TEST] Testing with local HuggingFace model")
    print("="*60)
    
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\nPyTorch not installed. Run: pip install torch")
        return False
    
    from benchmark import AuthorityBiasBenchmark, HuggingFaceLocalModel
    
    # Use a small model
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"\nLoading model: {model_name}")
    print("(This may take a few minutes on first run...)\n")
    
    try:
        model = HuggingFaceLocalModel(model=model_name, load_in_4bit=True)
        
        # Quick test
        print("\nQuick test generation...")
        response = model.generate("What is the capital of France?")
        print(f"Response: {response[:100]}...")
        
        # Run benchmark
        print("\nRunning benchmark...")
        benchmark = AuthorityBiasBenchmark(model=model)
        benchmark.run(verbose=True)
        benchmark.save_results("results_local.csv")
        benchmark.print_summary()
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("AUTHORITY BIAS BENCHMARK - TEST SUITE")
    print("="*60)
    print("\nSelect test mode:")
    print("1. Mock test (no API, no GPU - just verify pipeline)")
    print("2. HuggingFace API (free, needs HF token)")
    print("3. Local model (needs GPU or lots of RAM)")
    print("4. Run all tests")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        test_mock()
    elif choice == "2":
        test_hf_api()
    elif choice == "3":
        test_hf_local()
    elif choice == "4":
        test_mock()
        test_hf_api()
    else:
        print("Invalid choice. Running mock test...")
        test_mock()


if __name__ == "__main__":
    main()
