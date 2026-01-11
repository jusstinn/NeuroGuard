"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SafetyHullChart } from "@/components/visualizations/SafetyHullChart";
import { DegradationCurve } from "@/components/visualizations/DegradationCurve";
import { PlasticityCurve } from "@/components/visualizations/PlasticityCurve";
import { useNeuroGuardStore } from "@/lib/store";
import { Activity, Radar, TrendingDown, Zap } from "lucide-react";

export function VisualizationsPanel() {
  const { safetyHull, sandbaggingCurve, plasticityCurve, results } = useNeuroGuardStore();

  // Check if we have any data to display
  const hasSandbaggingData = sandbaggingCurve.length > 0;
  const hasPlasticityData = plasticityCurve.length > 0;
  const hasResults = Object.keys(results).length > 0;

  return (
    <Card className="border-border/50 h-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <Activity className="w-4 h-4 text-neuro-cyan" />
          Result Visualizations
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="hull" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="hull" className="text-xs">
              <Radar className="w-3 h-3 mr-1" />
              Safety Hull
            </TabsTrigger>
            <TabsTrigger value="degradation" className="text-xs">
              <TrendingDown className="w-3 h-3 mr-1" />
              Degradation
            </TabsTrigger>
            <TabsTrigger value="plasticity" className="text-xs">
              <Zap className="w-3 h-3 mr-1" />
              Plasticity
            </TabsTrigger>
          </TabsList>

          <TabsContent value="hull" className="mt-4">
            <SafetyHullChart data={safetyHull} />
          </TabsContent>

          <TabsContent value="degradation" className="mt-4">
            {hasSandbaggingData ? (
              <DegradationCurve data={sandbaggingCurve} />
            ) : (
              <EmptyState 
                title="No Sandbagging Data"
                description="Run the sandbagging evaluator to see the noise degradation curve."
              />
            )}
          </TabsContent>

          <TabsContent value="plasticity" className="mt-4">
            {hasPlasticityData ? (
              <PlasticityCurve data={plasticityCurve} />
            ) : (
              <EmptyState 
                title="No Plasticity Data"
                description="Run the plasticity evaluator to see the compliance curve."
              />
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex flex-col items-center justify-center h-64 text-center"
    >
      <div className="w-16 h-16 rounded-full bg-muted/30 flex items-center justify-center mb-4">
        <Activity className="w-8 h-8 text-muted-foreground" />
      </div>
      <h4 className="text-sm font-medium text-muted-foreground">{title}</h4>
      <p className="text-xs text-muted-foreground mt-1 max-w-xs">{description}</p>
    </motion.div>
  );
}
