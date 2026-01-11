"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Terminal, AlertCircle, Info, Bug } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useNeuroGuardStore, EvaluationLog as LogType } from "@/lib/store";

function LogIcon({ level }: { level: string }) {
  switch (level) {
    case "error":
      return <AlertCircle className="w-3 h-3 text-destructive" />;
    case "warning":
      return <AlertCircle className="w-3 h-3 text-risk-medium" />;
    case "debug":
      return <Bug className="w-3 h-3 text-muted-foreground" />;
    default:
      return <Info className="w-3 h-3 text-neuro-cyan" />;
  }
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function EvaluationLogPanel() {
  const { evaluationLogs } = useNeuroGuardStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [evaluationLogs]);

  return (
    <Card className="border-border/50 h-full flex flex-col">
      <CardHeader className="pb-2 flex-shrink-0">
        <CardTitle className="text-sm flex items-center gap-2">
          <Terminal className="w-4 h-4 text-neuro-cyan" />
          Live Evaluation Logs
          {evaluationLogs.length > 0 && (
            <span className="ml-auto text-xs font-mono text-muted-foreground">
              {evaluationLogs.length} entries
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden">
        <div
          ref={scrollRef}
          className="h-full overflow-y-auto space-y-1 font-mono text-xs pr-2"
        >
          {evaluationLogs.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              <p>Waiting for evaluation...</p>
            </div>
          ) : (
            <AnimatePresence mode="popLayout">
              {evaluationLogs.map((log, index) => (
                <motion.div
                  key={`${log.timestamp}-${index}`}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 10 }}
                  transition={{ duration: 0.2 }}
                  className="flex items-start gap-2 py-1 px-2 rounded hover:bg-muted/30"
                >
                  <LogIcon level={log.level} />
                  <span className="text-muted-foreground flex-shrink-0">
                    [{formatTimestamp(log.timestamp)}]
                  </span>
                  {log.evaluator && (
                    <span className="text-neuro-magenta flex-shrink-0">
                      [{log.evaluator}]
                    </span>
                  )}
                  <span className={
                    log.level === "error" 
                      ? "text-destructive" 
                      : log.level === "warning" 
                        ? "text-risk-medium" 
                        : "text-foreground"
                  }>
                    {log.message}
                  </span>
                </motion.div>
              ))}
            </AnimatePresence>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
