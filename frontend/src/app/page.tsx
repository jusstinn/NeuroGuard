"use client";

import { motion } from "framer-motion";
import { Sidebar } from "@/components/dashboard/Sidebar";
import { EvaluatorCard } from "@/components/dashboard/EvaluatorCard";
import { EvaluationLogPanel } from "@/components/dashboard/EvaluationLog";
import { VisualizationsPanel } from "@/components/dashboard/VisualizationsPanel";
import { Button } from "@/components/ui/button";
import { useNeuroGuardStore } from "@/lib/store";
import { Play, RotateCcw, Shield } from "lucide-react";

const evaluators = [
  {
    id: "authority_bias",
    name: "🏆 Authority Bias",
    description: "Featured: Tests resistance to fake expert pressure",
    icon: "shield" as const,
    featured: true,
  },
  {
    id: "sandbagging",
    name: "Sandbagging Detection",
    description: "Noise injection to detect strategic underperformance",
    icon: "shield" as const,
  },
  {
    id: "sycophancy",
    name: "Sycophancy Analysis",
    description: "Multi-turn pressure testing for bias agreement",
    icon: "heart" as const,
  },
  {
    id: "dark_patterns",
    name: "Dark Pattern Recon",
    description: "Deceptive UI/UX generation detection",
    icon: "eye" as const,
  },
  {
    id: "plasticity",
    name: "Malicious Plasticity",
    description: "In-context learning vulnerability measurement",
    icon: "zap" as const,
  },
];

export default function Dashboard() {
  const { modelLoaded, reset, safetyHull, results } = useNeuroGuardStore();
  const hasResults = Object.keys(results).length > 0;

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="border-b border-border bg-card/50 backdrop-blur-sm px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold">Evaluation Dashboard</h2>
              <p className="text-sm text-muted-foreground">
                Run comprehensive safety evaluations on your fine-tuned LLM
              </p>
            </div>
            <div className="flex items-center gap-2">
              {hasResults && (
                <Button variant="outline" size="sm" onClick={reset}>
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Reset
                </Button>
              )}
              <Button 
                variant="neuro" 
                size="sm" 
                disabled={!modelLoaded}
                onClick={() => {
                  // Run all evaluators
                }}
              >
                <Play className="w-4 h-4 mr-2" />
                Run Full Suite
              </Button>
            </div>
          </div>
        </header>

        {/* Dashboard Grid */}
        <div className="flex-1 overflow-auto p-6">
          <div className="grid grid-cols-12 gap-6 h-full min-h-[800px]">
            {/* Evaluator Cards - Left Column */}
            <div className="col-span-12 lg:col-span-4 space-y-4">
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ staggerChildren: 0.1 }}
                className="space-y-4"
              >
                {evaluators.map((evaluator, index) => (
                  <motion.div
                    key={evaluator.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <EvaluatorCard {...evaluator} />
                  </motion.div>
                ))}
              </motion.div>

              {/* Overall Safety Score */}
              {safetyHull && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 rounded-xl border border-border bg-gradient-to-br from-neuro-cyan/10 to-neuro-magenta/10"
                >
                  <div className="flex items-center gap-3">
                    <div className="p-3 rounded-lg bg-background/50">
                      <Shield className="w-6 h-6 text-neuro-cyan" />
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Overall Safety Score</p>
                      <p className="text-3xl font-bold gradient-text">
                        {(safetyHull.overall * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>

            {/* Center Column - Visualizations */}
            <div className="col-span-12 lg:col-span-5">
              <VisualizationsPanel />
            </div>

            {/* Right Column - Logs */}
            <div className="col-span-12 lg:col-span-3 h-[600px]">
              <EvaluationLogPanel />
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="border-t border-border bg-card/50 px-6 py-3">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              NeuroGuard v1.0.0 • AI Safety Evaluation Platform
            </span>
            <span className="font-mono">
              {modelLoaded ? "● Model Active" : "○ No Model"}
            </span>
          </div>
        </footer>
      </main>
    </div>
  );
}
