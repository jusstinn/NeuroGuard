"""
NeuroGuard Backend API
======================
FastAPI server for AI safety evaluation platform.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from core.model_manager import ModelManager, ModelConfig
from evaluators import (
    SandbaggingEvaluator,
    SycophancyEvaluator,
    DarkPatternEvaluator,
    PlasticityEvaluator,
    AuthorityBiasEvaluator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
model_manager: Optional[ModelManager] = None
evaluators: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global model_manager, evaluators
    
    logger.info("Initializing NeuroGuard...")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Initialize evaluators
    evaluators = {
        "sandbagging": SandbaggingEvaluator(),
        "sycophancy": SycophancyEvaluator(),
        "dark_patterns": DarkPatternEvaluator(),
        "plasticity": PlasticityEvaluator(),
        "authority_bias": AuthorityBiasEvaluator(),  # 🏆 Featured Benchmark
    }
    
    logger.info("NeuroGuard initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down NeuroGuard...")
    if model_manager and model_manager.current_model:
        model_manager.unload_model()


app = FastAPI(
    title="NeuroGuard API",
    description="AI Safety Evaluation Platform - Detect sandbagging, sycophancy, dark patterns, and malicious plasticity",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Pydantic Models ============

class ModelLoadRequest(BaseModel):
    """Request to load a model."""
    model_path: str = Field(..., description="HuggingFace model path or local path")
    adapter_path: Optional[str] = Field(None, description="LoRA adapter path")
    load_in_4bit: bool = Field(True, description="Load with 4-bit quantization")
    load_in_8bit: bool = Field(False, description="Load with 8-bit quantization")
    temperature: float = Field(0.7, ge=0, le=2)
    max_new_tokens: int = Field(512, ge=1, le=4096)


class EvaluationRequest(BaseModel):
    """Request to run an evaluation."""
    config: Dict[str, Any] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    """Request to generate text."""
    prompt: str
    temperature: Optional[float] = None
    max_new_tokens: Optional[int] = None


# ============ API Endpoints ============

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "NeuroGuard API",
        "version": "1.0.0",
        "status": "operational",
        "evaluators": list(evaluators.keys()),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.current_model is not None if model_manager else False,
    }


# ============ Model Management ============

@app.post("/api/model/load")
async def load_model(request: ModelLoadRequest):
    """Load a model for evaluation."""
    try:
        config = ModelConfig(
            model_path=request.model_path,
            adapter_path=request.adapter_path,
            load_in_4bit=request.load_in_4bit,
            load_in_8bit=request.load_in_8bit,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
        )
        
        model_manager.load_model(config)
        
        return {
            "status": "success",
            "message": f"Model loaded: {request.model_path}",
            "info": model_manager.get_model_info(),
        }
    except Exception as e:
        logger.exception("Failed to load model")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/unload")
async def unload_model():
    """Unload the current model."""
    try:
        model_manager.unload_model()
        return {"status": "success", "message": "Model unloaded"}
    except Exception as e:
        logger.exception("Failed to unload model")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    return model_manager.get_model_info() if model_manager else {"loaded": False}


@app.post("/api/model/generate")
async def generate_text(request: GenerateRequest):
    """Generate text from the loaded model."""
    if not model_manager or not model_manager.current_model:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        response = model_manager.generate(
            request.prompt,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
        )
        return {"response": response}
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Evaluator Info ============

@app.get("/api/evaluators")
async def list_evaluators():
    """List all available evaluators."""
    return {
        name: evaluator.get_info()
        for name, evaluator in evaluators.items()
    }


@app.get("/api/evaluators/{evaluator_name}")
async def get_evaluator_info(evaluator_name: str):
    """Get information about a specific evaluator."""
    if evaluator_name not in evaluators:
        raise HTTPException(status_code=404, detail=f"Evaluator not found: {evaluator_name}")
    
    return evaluators[evaluator_name].get_info()


# ============ Evaluation Endpoints ============

@app.post("/api/eval/{evaluator_name}")
async def run_evaluation(evaluator_name: str, request: EvaluationRequest):
    """Run an evaluation (non-streaming)."""
    if evaluator_name not in evaluators:
        raise HTTPException(status_code=404, detail=f"Evaluator not found: {evaluator_name}")
    
    if not model_manager or not model_manager.current_model:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        evaluator = evaluators[evaluator_name]
        result = await evaluator.evaluate(model_manager, request.config)
        return result.to_dict()
    except Exception as e:
        logger.exception(f"Evaluation failed: {evaluator_name}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/eval/{evaluator_name}/stream")
async def stream_evaluation(evaluator_name: str):
    """Run an evaluation with SSE streaming."""
    if evaluator_name not in evaluators:
        raise HTTPException(status_code=404, detail=f"Evaluator not found: {evaluator_name}")
    
    if not model_manager or not model_manager.current_model:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    evaluator = evaluators[evaluator_name]
    
    async def event_generator():
        try:
            async for event in evaluator.evaluate_stream(model_manager, {}):
                yield event.to_sse()
        except Exception as e:
            logger.exception(f"Streaming evaluation failed: {evaluator_name}")
            yield f'data: {{"type": "error", "message": "{str(e)}"}}\n\n'
    
    return EventSourceResponse(event_generator())


@app.post("/api/eval/{evaluator_name}/stream")
async def stream_evaluation_with_config(evaluator_name: str, request: EvaluationRequest):
    """Run an evaluation with SSE streaming and custom config."""
    if evaluator_name not in evaluators:
        raise HTTPException(status_code=404, detail=f"Evaluator not found: {evaluator_name}")
    
    if not model_manager or not model_manager.current_model:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    evaluator = evaluators[evaluator_name]
    
    async def event_generator():
        try:
            async for event in evaluator.evaluate_stream(model_manager, request.config):
                yield event.to_sse()
        except Exception as e:
            logger.exception(f"Streaming evaluation failed: {evaluator_name}")
            yield f'data: {{"type": "error", "message": "{str(e)}"}}\n\n'
    
    return EventSourceResponse(event_generator())


# ============ Full Evaluation Suite ============

@app.post("/api/eval/full")
async def run_full_evaluation(request: EvaluationRequest):
    """Run all evaluators and return combined results."""
    if not model_manager or not model_manager.current_model:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    results = {}
    
    for name, evaluator in evaluators.items():
        try:
            result = await evaluator.evaluate(model_manager, request.config.get(name, {}))
            results[name] = result.to_dict()
        except Exception as e:
            logger.exception(f"Evaluation failed: {name}")
            results[name] = {"error": str(e), "status": "failed"}
    
    # Calculate overall safety hull
    safety_hull = calculate_safety_hull(results)
    
    return {
        "results": results,
        "safety_hull": safety_hull,
        "model_info": model_manager.get_model_info(),
    }


@app.get("/api/eval/full/stream")
async def stream_full_evaluation():
    """Stream all evaluations sequentially."""
    if not model_manager or not model_manager.current_model:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    async def event_generator():
        results = {}
        
        for name, evaluator in evaluators.items():
            yield f'data: {{"type": "evaluator_start", "evaluator": "{name}"}}\n\n'
            
            try:
                async for event in evaluator.evaluate_stream(model_manager, {}):
                    # Prefix event with evaluator name
                    event.data["evaluator"] = name
                    yield event.to_sse()
                
                # Get final result
                result = await evaluator.evaluate(model_manager, {})
                results[name] = result.to_dict()
                
            except Exception as e:
                logger.exception(f"Streaming evaluation failed: {name}")
                yield f'data: {{"type": "evaluator_error", "evaluator": "{name}", "error": "{str(e)}"}}\n\n'
                results[name] = {"error": str(e), "status": "failed"}
        
        # Final combined result
        safety_hull = calculate_safety_hull(results)
        yield f'data: {{"type": "final_result", "safety_hull": {safety_hull}, "results": "complete"}}\n\n'
    
    return EventSourceResponse(event_generator())


def calculate_safety_hull(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate the safety hull - a summary across all evaluation axes.
    
    Returns scores for:
    - Robustness (from sandbagging - resistance to noise)
    - Independence (from sycophancy - not agreeing with false beliefs)
    - Ethics (from dark_patterns - not generating deceptive UI)
    - Rigidity (from plasticity - resistance to jailbreaks)
    """
    hull = {
        "robustness": 0.5,
        "independence": 0.5,
        "ethics": 0.5,
        "rigidity": 0.5,
    }
    
    if "sandbagging" in results and "score" in results["sandbagging"]:
        hull["robustness"] = results["sandbagging"]["score"]
    
    if "sycophancy" in results and "score" in results["sycophancy"]:
        hull["independence"] = results["sycophancy"]["score"]
    
    if "dark_patterns" in results and "score" in results["dark_patterns"]:
        hull["ethics"] = results["dark_patterns"]["score"]
    
    if "plasticity" in results and "score" in results["plasticity"]:
        hull["rigidity"] = results["plasticity"]["score"]
    
    # Overall safety score
    hull["overall"] = sum(hull.values()) / 4
    
    return hull


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
