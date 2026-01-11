"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { 
  Play, 
  Loader2, 
  Shield, 
  Heart, 
  Eye, 
  Zap,
  CheckCircle,
  XCircle
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useNeuroGuardStore, EvaluationResult } from "@/lib/store";
import { streamEvaluation } from "@/lib/api";
import { cn, formatScore, getRiskColor, getRiskBgColor } from "@/lib/utils";

interface EvaluatorCardProps {
  id: string;
  name: string;
  description: string;
  icon: "shield" | "heart" | "eye" | "zap";
}

const iconMap = {
  shield: Shield,
  heart: Heart,
  eye: Eye,
  zap: Zap,
};

export function EvaluatorCard({ id, name, description, icon }: EvaluatorCardProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  
  const { 
    modelLoaded, 
    results, 
    setResult, 
    addLog,
    updateSandbaggingCurve,
    updatePlasticityCurve,
  } = useNeuroGuardStore();

  const result = results[id];
  const Icon = iconMap[icon];

  const handleRun = () => {
    if (!modelLoaded || isRunning) return;

    setIsRunning(true);
    setProgress(0);

    const cleanup = streamEvaluation(
      id,
      (event: unknown) => {
        const data = event as Record<string, unknown>;
        
        // Handle different event types
        switch (data.type) {
          case "log":
            addLog({
              timestamp: data.timestamp as string || new Date().toISOString(),
              level: (data.level as string) || "info",
              message: data.message as string,
              evaluator: id,
            });
            break;
          
          case "progress":
            const current = data.current as number;
            const total = data.total as number;
            setProgress((current / total) * 100);
            break;
          
          case "curve_update":
            if (id === "sandbagging" && data.performance_curve) {
              updateSandbaggingCurve(data.performance_curve as Array<{ sigma: number; mean_accuracy: number }>);
            }
            if (id === "plasticity" && data.compliance_curve) {
              updatePlasticityCurve(data.compliance_curve as Array<{ shot_level: number; compliance_rate: number }>);
            }
            break;
          
          case "result":
            setResult(id, {
              evaluator_name: id,
              status: "completed",
              score: data.score as number,
              risk_level: data.risk_level as string,
              metrics: (data.metrics as Record<string, number>) 
                ? Object.entries(data.metrics).map(([name, value]) => ({
                    name,
                    value: value as number,
                  }))
                : [],
              raw_data: data as Record<string, unknown>,
              logs: [],
            });
            setIsRunning(false);
            setProgress(100);
            break;
          
          case "error":
            addLog({
              timestamp: new Date().toISOString(),
              level: "error",
              message: data.message as string,
              evaluator: id,
            });
            setIsRunning(false);
            break;
        }
      },
      (error) => {
        addLog({
          timestamp: new Date().toISOString(),
          level: "error",
          message: error.message,
          evaluator: id,
        });
        setIsRunning(false);
      },
      () => {
        setIsRunning(false);
      }
    );

    // Cleanup on unmount would go here in a useEffect
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className={cn(
        "border-border/50 transition-all duration-300",
        isRunning && "border-neuro-cyan/50 shadow-[0_0_20px_rgba(0,217,255,0.1)]",
        result && result.status === "completed" && "border-risk-low/30"
      )}>
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              <div className={cn(
                "p-2 rounded-lg",
                result 
                  ? getRiskBgColor(result.risk_level)
                  : "bg-muted"
              )}>
                <Icon className={cn(
                  "w-4 h-4",
                  result 
                    ? getRiskColor(result.risk_level)
                    : "text-muted-foreground"
                )} />
              </div>
              <div>
                <CardTitle className="text-sm">{name}</CardTitle>
                <CardDescription className="text-xs">{description}</CardDescription>
              </div>
            </div>
            {result && (
              <div className={cn(
                "text-2xl font-bold font-mono",
                getRiskColor(result.risk_level)
              )}>
                {formatScore(result.score)}
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Progress bar when running */}
          {isRunning && (
            <div className="space-y-1">
              <Progress value={progress} className="h-1" />
              <p className="text-xs text-muted-foreground text-center">
                {progress.toFixed(0)}% complete
              </p>
            </div>
          )}

          {/* Results */}
          {result && result.metrics && result.metrics.length > 0 && (
            <div className="grid grid-cols-2 gap-2">
              {result.metrics.slice(0, 4).map((metric, idx) => (
                <div key={idx} className="text-xs">
                  <span className="text-muted-foreground">{metric.name}:</span>
                  <span className="ml-1 font-mono">
                    {typeof metric.value === "number" 
                      ? metric.value.toFixed(2) 
                      : metric.value}
                    {metric.unit}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Run button */}
          <Button
            onClick={handleRun}
            disabled={!modelLoaded || isRunning}
            variant={result ? "outline" : "glow"}
            size="sm"
            className="w-full"
          >
            {isRunning ? (
              <>
                <Loader2 className="w-3 h-3 mr-2 animate-spin" />
                Running...
              </>
            ) : result ? (
              <>
                <CheckCircle className="w-3 h-3 mr-2" />
                Re-run
              </>
            ) : (
              <>
                <Play className="w-3 h-3 mr-2" />
                Run Evaluation
              </>
            )}
          </Button>
        </CardContent>
      </Card>
    </motion.div>
  );
}
