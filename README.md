# NeuroGuard 🧠🛡️

**Next-Generation AI Safety Evaluation Platform**

NeuroGuard is a comprehensive evaluation framework for detecting safety vulnerabilities in fine-tuned LLMs. Upload your model (LoRA/PEFT/Full weights) and run a battery of adversarial evaluations to detect:

- **Sandbagging**: Strategic underperformance detected via noise injection
- **Sycophancy**: Tendency to agree with user biases over factual truth
- **Dark Patterns**: Deceptive UI/UX generation capabilities
- **Malicious Plasticity**: In-context learning vulnerability to harmful examples

![NeuroGuard Dashboard](docs/dashboard-preview.png)

## 🔬 Research Background

### Sandbagging Detection
*Reference: "Sandbag Detection through Model Degradation" (Apart Research) & "LLMs Can Covertly Sandbag" (2025)*

The noise injection methodology exploits the "fragility" of sandbagging models. A model faking incompetence maintains its facade through precise weight configurations - noise disrupts this precision, causing faster/chaotic degradation compared to genuinely limited models.

### Sycophancy Quantification
*Reference: "Measuring Sycophancy in Multi-turn Dialogues" (2025)*

Multi-turn pressure testing measures how quickly models abandon factual truth to please users. The Flip Rate and Flip Velocity metrics quantify this dangerous behavior.

### Dark Pattern Detection
*Reference: "Hidden Darkness in LLM-Generated Designs" (2025)*

Analyzes LLM-generated UI code for deceptive design patterns including confirmshaming, false urgency, forced action, and hidden costs.

### Malicious Plasticity
*Reference: "Narrow in-context examples can produce broadly misaligned LLMs" (2025)*

Measures the Compliance Velocity (Vc) - how quickly a model's safety degrades when exposed to harmful in-context examples:

$$V_c = \frac{\Delta \text{Compliance Rate}}{\Delta \text{Examples}}$$

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (recommended)

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to access the dashboard.

## 📊 Architecture

```
NeuroGuard/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── core/
│   │   └── model_manager.py # HuggingFace model loading/management
│   └── evaluators/
│       ├── base.py          # Abstract evaluator interface
│       ├── sandbagging.py   # Noise injection evaluator
│       ├── sycophancy.py    # Multi-turn pressure testing
│       ├── dark_patterns.py # UI code analysis
│       └── plasticity.py    # ICL vulnerability testing
│
├── frontend/
│   ├── src/
│   │   ├── app/             # Next.js 14 App Router
│   │   ├── components/      # React components
│   │   │   ├── ui/          # Shadcn/UI components
│   │   │   ├── dashboard/   # Dashboard-specific components
│   │   │   └── visualizations/
│   │   │       ├── SafetyHullChart.tsx
│   │   │       ├── DegradationCurve.tsx
│   │   │       └── PlasticityCurve.tsx
│   │   └── lib/
│   │       ├── store.ts     # Zustand state management
│   │       └── api.ts       # API client with SSE support
│   └── ...
└── README.md
```

## 🔌 API Endpoints

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/model/load` | POST | Load a model for evaluation |
| `/api/model/unload` | POST | Unload the current model |
| `/api/model/info` | GET | Get loaded model information |

### Evaluation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/evaluators` | GET | List all available evaluators |
| `/api/eval/{name}` | POST | Run evaluation (blocking) |
| `/api/eval/{name}/stream` | GET/POST | Run evaluation with SSE streaming |
| `/api/eval/full` | POST | Run all evaluators |
| `/api/eval/full/stream` | GET | Stream all evaluations |

## 📈 The Safety Hull

The Safety Hull is a radar chart visualization summarizing the model's performance across four safety axes:

- **Robustness**: Resistance to noise injection (sandbagging score)
- **Independence**: Resistance to user pressure (inverse sycophancy)
- **Ethics**: Avoidance of dark patterns
- **Rigidity**: Resistance to jailbreak examples (inverse plasticity)

## 🛠️ Technical Stack

### Backend
- **FastAPI** - High-performance async API framework
- **PyTorch** - Deep learning framework
- **Transformers** - HuggingFace model loading
- **PEFT** - Parameter-efficient fine-tuning support
- **BitsAndBytes** - Quantization (4-bit/8-bit)
- **SSE-Starlette** - Server-Sent Events for streaming

### Frontend
- **Next.js 14** - React framework with App Router
- **TailwindCSS** - Utility-first CSS
- **Shadcn/UI** - Beautiful component library
- **Framer Motion** - Animation library
- **Recharts** - Data visualization
- **Zustand** - Lightweight state management
- **TanStack Query** - Data fetching

## 🎨 Design Philosophy

NeuroGuard's UI is "Research-Grade Beautiful" - clinical precision meets aesthetic excellence:

- **Dark mode default** for reduced eye strain during long evaluation sessions
- **Monospace fonts** for metrics and technical data
- **Subtle gradients** inspired by Anthropic's research publications
- **Real-time streaming** for live evaluation feedback
- **Information density** optimized for researcher workflows

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📚 Citations

If you use NeuroGuard in your research, please cite:

```bibtex
@software{neuroguard2025,
  title={NeuroGuard: AI Safety Evaluation Platform},
  year={2025},
  url={https://github.com/your-org/neuroguard}
}
```

---

Built with 🧠 for AI Safety Research
