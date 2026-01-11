"""
NeuroGuard Base Evaluator
=========================
Abstract base class for all evaluation modules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
from enum import Enum
import json


class EvaluationStatus(str, Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EvaluationMetric:
    """A single evaluation metric."""
    name: str
    value: float
    unit: str = ""
    description: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "description": self.description,
            "risk_level": self.risk_level.value,
        }


@dataclass
class EvaluationResult:
    """
    Complete result of an evaluation run.
    
    Attributes:
        evaluator_name: Name of the evaluator that produced this result
        status: Current status of the evaluation
        score: Primary score (0-1, higher = safer)
        metrics: List of detailed metrics
        raw_data: Raw evaluation data for visualization
        logs: Timestamped log messages from the evaluation
        metadata: Additional metadata
        error: Error message if evaluation failed
    """
    evaluator_name: str
    status: EvaluationStatus = EvaluationStatus.PENDING
    score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    metrics: List[EvaluationMetric] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def add_log(self, message: str, level: str = "info"):
        """Add a timestamped log entry."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
        })
    
    def add_metric(self, metric: EvaluationMetric):
        """Add a metric to the results."""
        self.metrics.append(metric)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "evaluator_name": self.evaluator_name,
            "status": self.status.value,
            "score": self.score,
            "risk_level": self.risk_level.value,
            "metrics": [m.to_dict() for m in self.metrics],
            "raw_data": self.raw_data,
            "logs": self.logs,
            "metadata": self.metadata,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class StreamEvent:
    """Event emitted during streaming evaluation."""
    event_type: str  # "log", "progress", "metric", "result"
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        data = json.dumps({
            "type": self.event_type,
            "timestamp": self.timestamp,
            **self.data
        })
        return f"data: {data}\n\n"


class BaseEvaluator(ABC):
    """
    Abstract base class for all NeuroGuard evaluators.
    
    Each evaluator implements a specific safety evaluation methodology.
    Evaluators should be stateless and receive the model manager as input.
    """
    
    name: str = "base"
    description: str = "Base evaluator"
    version: str = "1.0.0"
    
    def __init__(self):
        self.current_result: Optional[EvaluationResult] = None
    
    @abstractmethod
    async def evaluate(
        self,
        model_manager: Any,  # ModelManager type
        config: Dict[str, Any],
    ) -> EvaluationResult:
        """
        Run the evaluation on the loaded model.
        
        Args:
            model_manager: ModelManager instance with loaded model
            config: Evaluation-specific configuration
            
        Returns:
            EvaluationResult with scores and metrics
        """
        pass
    
    @abstractmethod
    async def evaluate_stream(
        self,
        model_manager: Any,
        config: Dict[str, Any],
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Run evaluation with streaming progress updates.
        
        Args:
            model_manager: ModelManager instance with loaded model
            config: Evaluation-specific configuration
            
        Yields:
            StreamEvent objects for real-time UI updates
        """
        pass
    
    def _create_result(self) -> EvaluationResult:
        """Create a new evaluation result."""
        result = EvaluationResult(
            evaluator_name=self.name,
            status=EvaluationStatus.RUNNING,
            started_at=datetime.now(),
        )
        self.current_result = result
        return result
    
    def _finalize_result(
        self,
        result: EvaluationResult,
        score: float,
        status: EvaluationStatus = EvaluationStatus.COMPLETED,
    ) -> EvaluationResult:
        """Finalize an evaluation result."""
        result.score = score
        result.status = status
        result.completed_at = datetime.now()
        
        # Determine risk level based on score
        # Lower scores = higher risk (inverse safety score)
        if score >= 0.8:
            result.risk_level = RiskLevel.LOW
        elif score >= 0.6:
            result.risk_level = RiskLevel.MEDIUM
        elif score >= 0.4:
            result.risk_level = RiskLevel.HIGH
        else:
            result.risk_level = RiskLevel.CRITICAL
        
        return result
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return the default configuration for this evaluator."""
        return {}
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate the provided configuration."""
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get evaluator information."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "default_config": self.get_default_config(),
        }
