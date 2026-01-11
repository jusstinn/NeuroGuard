"use client";

import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { motion } from "framer-motion";
import { SafetyHull } from "@/lib/store";

interface SafetyHullChartProps {
  data: SafetyHull | null;
}

export function SafetyHullChart({ data }: SafetyHullChartProps) {
  const chartData = data
    ? [
        { axis: "Robustness", value: data.robustness * 100, fullMark: 100 },
        { axis: "Independence", value: data.independence * 100, fullMark: 100 },
        { axis: "Ethics", value: data.ethics * 100, fullMark: 100 },
        { axis: "Rigidity", value: data.rigidity * 100, fullMark: 100 },
      ]
    : [
        { axis: "Robustness", value: 50, fullMark: 100 },
        { axis: "Independence", value: 50, fullMark: 100 },
        { axis: "Ethics", value: 50, fullMark: 100 },
        { axis: "Rigidity", value: 50, fullMark: 100 },
      ];

  const overallScore = data ? data.overall * 100 : 50;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="relative"
    >
      {/* Overall Score Badge */}
      <div className="absolute top-0 right-0 z-10">
        <div className="flex flex-col items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-neuro-cyan/20 to-neuro-magenta/20 border border-neuro-cyan/30">
          <span className="text-2xl font-bold text-neuro-cyan">
            {overallScore.toFixed(0)}
          </span>
          <span className="text-xs text-muted-foreground">Overall</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={350}>
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={chartData}>
          <PolarGrid
            stroke="rgba(255,255,255,0.1)"
            strokeDasharray="3 3"
          />
          <PolarAngleAxis
            dataKey="axis"
            tick={{ fill: "#a3a3a3", fontSize: 12 }}
            tickLine={false}
          />
          <PolarRadiusAxis
            angle={30}
            domain={[0, 100]}
            tick={{ fill: "#525252", fontSize: 10 }}
            tickCount={5}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(0 0% 6%)",
              border: "1px solid hsl(0 0% 14%)",
              borderRadius: "8px",
              boxShadow: "0 0 20px rgba(0, 217, 255, 0.2)",
            }}
            formatter={(value: number) => [`${value.toFixed(1)}%`, "Score"]}
          />
          <Radar
            name="Safety Hull"
            dataKey="value"
            stroke="#00D9FF"
            fill="url(#radarGradient)"
            fillOpacity={0.4}
            strokeWidth={2}
          />
          <defs>
            <linearGradient id="radarGradient" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#00D9FF" stopOpacity={0.6} />
              <stop offset="100%" stopColor="#FF006E" stopOpacity={0.6} />
            </linearGradient>
          </defs>
        </RadarChart>
      </ResponsiveContainer>

      {/* Axis Labels with descriptions */}
      <div className="grid grid-cols-2 gap-4 mt-4 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-neuro-cyan" />
          <span className="text-muted-foreground">
            <strong className="text-foreground">Robustness:</strong> Sandbagging resistance
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-neuro-magenta" />
          <span className="text-muted-foreground">
            <strong className="text-foreground">Independence:</strong> Sycophancy resistance
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-neuro-lime" />
          <span className="text-muted-foreground">
            <strong className="text-foreground">Ethics:</strong> Dark pattern avoidance
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-neuro-amber" />
          <span className="text-muted-foreground">
            <strong className="text-foreground">Rigidity:</strong> Jailbreak resistance
          </span>
        </div>
      </div>
    </motion.div>
  );
}
