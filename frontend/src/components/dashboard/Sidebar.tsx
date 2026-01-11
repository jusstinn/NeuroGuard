"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { 
  Brain, 
  Upload, 
  Settings, 
  Loader2,
  Check,
  AlertCircle
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useNeuroGuardStore } from "@/lib/store";
import { loadModel, unloadModel } from "@/lib/api";

export function Sidebar() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const { 
    modelConfig, 
    setModelConfig, 
    modelLoaded, 
    setModelLoaded,
    modelInfo 
  } = useNeuroGuardStore();

  const handleLoadModel = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await loadModel({
        model_path: modelConfig.modelPath,
        adapter_path: modelConfig.adapterPath || undefined,
        load_in_4bit: modelConfig.loadIn4bit,
        load_in_8bit: modelConfig.loadIn8bit,
        temperature: modelConfig.temperature,
        max_new_tokens: modelConfig.maxNewTokens,
      });
      
      setModelLoaded(true, (response as { info: Record<string, unknown> }).info);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load model");
    } finally {
      setIsLoading(false);
    }
  };

  const handleUnloadModel = async () => {
    setIsLoading(true);
    try {
      await unloadModel();
      setModelLoaded(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to unload model");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <aside className="w-80 border-r border-border bg-card/50 backdrop-blur-sm p-4 flex flex-col gap-4 overflow-y-auto">
      {/* Logo */}
      <div className="flex items-center gap-3 mb-4">
        <div className="relative">
          <Brain className="w-8 h-8 text-neuro-cyan" />
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-neuro-magenta rounded-full animate-pulse" />
        </div>
        <div>
          <h1 className="text-xl font-bold gradient-text">NeuroGuard</h1>
          <p className="text-xs text-muted-foreground">AI Safety Evaluation</p>
        </div>
      </div>

      {/* Model Status */}
      <Card className="border-border/50">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${modelLoaded ? 'bg-risk-low' : 'bg-muted-foreground'}`} />
            Model Status
          </CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground">
          {modelLoaded ? (
            <div className="space-y-1">
              <p className="text-risk-low font-medium">Model Loaded</p>
              {modelInfo && (
                <>
                  <p>Parameters: {((modelInfo as { total_parameters?: number }).total_parameters || 0).toLocaleString()}</p>
                  <p>Device: {(modelInfo as { device?: string }).device}</p>
                  <p>Quantized: {(modelInfo as { quantized?: boolean }).quantized ? "Yes" : "No"}</p>
                </>
              )}
            </div>
          ) : (
            <p>No model loaded</p>
          )}
        </CardContent>
      </Card>

      {/* Model Configuration */}
      <Card className="border-border/50">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Model Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Model Path */}
          <div className="space-y-2">
            <Label htmlFor="modelPath" className="text-xs">Model Path</Label>
            <Input
              id="modelPath"
              placeholder="meta-llama/Llama-2-7b-chat-hf"
              value={modelConfig.modelPath}
              onChange={(e) => setModelConfig({ modelPath: e.target.value })}
              className="text-xs font-mono"
            />
          </div>

          {/* Adapter Path */}
          <div className="space-y-2">
            <Label htmlFor="adapterPath" className="text-xs">LoRA Adapter (optional)</Label>
            <Input
              id="adapterPath"
              placeholder="path/to/adapter"
              value={modelConfig.adapterPath}
              onChange={(e) => setModelConfig({ adapterPath: e.target.value })}
              className="text-xs font-mono"
            />
          </div>

          {/* Quantization */}
          <div className="flex items-center justify-between">
            <Label htmlFor="load4bit" className="text-xs">4-bit Quantization</Label>
            <Switch
              id="load4bit"
              checked={modelConfig.loadIn4bit}
              onCheckedChange={(checked) => setModelConfig({ loadIn4bit: checked, loadIn8bit: false })}
            />
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="load8bit" className="text-xs">8-bit Quantization</Label>
            <Switch
              id="load8bit"
              checked={modelConfig.loadIn8bit}
              onCheckedChange={(checked) => setModelConfig({ loadIn8bit: checked, loadIn4bit: false })}
            />
          </div>

          {/* Temperature */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">Temperature</Label>
              <span className="text-xs font-mono text-muted-foreground">
                {modelConfig.temperature.toFixed(2)}
              </span>
            </div>
            <Slider
              value={[modelConfig.temperature]}
              onValueChange={([value]) => setModelConfig({ temperature: value })}
              min={0}
              max={2}
              step={0.1}
            />
          </div>

          {/* Max Tokens */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">Max New Tokens</Label>
              <span className="text-xs font-mono text-muted-foreground">
                {modelConfig.maxNewTokens}
              </span>
            </div>
            <Slider
              value={[modelConfig.maxNewTokens]}
              onValueChange={([value]) => setModelConfig({ maxNewTokens: value })}
              min={64}
              max={2048}
              step={64}
            />
          </div>

          {/* Error */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-2 p-2 bg-destructive/10 rounded-md text-destructive text-xs"
            >
              <AlertCircle className="w-4 h-4" />
              {error}
            </motion.div>
          )}

          {/* Load/Unload Button */}
          <Button
            onClick={modelLoaded ? handleUnloadModel : handleLoadModel}
            disabled={isLoading}
            variant={modelLoaded ? "outline" : "neuro"}
            className="w-full"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : modelLoaded ? (
              <Check className="w-4 h-4 mr-2" />
            ) : (
              <Upload className="w-4 h-4 mr-2" />
            )}
            {isLoading ? "Processing..." : modelLoaded ? "Unload Model" : "Load Model"}
          </Button>
        </CardContent>
      </Card>

      {/* Info */}
      <div className="mt-auto p-3 bg-muted/30 rounded-lg border border-border/50">
        <p className="text-xs text-muted-foreground">
          NeuroGuard evaluates fine-tuned LLMs for safety vulnerabilities including 
          sandbagging, sycophancy, dark patterns, and malicious plasticity.
        </p>
      </div>
    </aside>
  );
}
