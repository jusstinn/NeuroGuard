"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { motion } from "framer-motion";

interface DegradationCurveProps {
  data: Array<{ sigma: number; mean_accuracy: number }>;
  title?: string;
}

export function DegradationCurve({ data, title = "Performance vs. Noise Injection" }: DegradationCurveProps) {
  const formattedData = data.map((d) => ({
    sigma: d.sigma,
    accuracy: d.mean_accuracy * 100,
  }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full"
    >
      <h4 className="text-sm font-medium mb-4 text-muted-foreground">{title}</h4>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart
          data={formattedData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis
            dataKey="sigma"
            stroke="#525252"
            tick={{ fill: "#a3a3a3", fontSize: 11 }}
            label={{
              value: "Noise σ",
              position: "insideBottomRight",
              offset: -5,
              fill: "#a3a3a3",
              fontSize: 11,
            }}
          />
          <YAxis
            stroke="#525252"
            tick={{ fill: "#a3a3a3", fontSize: 11 }}
            domain={[0, 100]}
            label={{
              value: "Accuracy %",
              angle: -90,
              position: "insideLeft",
              fill: "#a3a3a3",
              fontSize: 11,
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(0 0% 6%)",
              border: "1px solid hsl(0 0% 14%)",
              borderRadius: "8px",
              boxShadow: "0 0 20px rgba(0, 217, 255, 0.2)",
            }}
            formatter={(value: number) => [`${value.toFixed(1)}%`, "Accuracy"]}
            labelFormatter={(label) => `σ = ${label}`}
          />
          <ReferenceLine
            y={50}
            stroke="#FF006E"
            strokeDasharray="5 5"
            strokeOpacity={0.5}
          />
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke="#00D9FF"
            strokeWidth={2}
            dot={{ fill: "#00D9FF", strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6, fill: "#00D9FF", stroke: "#fff", strokeWidth: 2 }}
          />
          <defs>
            <linearGradient id="lineGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#00D9FF" />
              <stop offset="100%" stopColor="#FF006E" />
            </linearGradient>
          </defs>
        </LineChart>
      </ResponsiveContainer>
      <div className="flex items-center justify-center gap-4 mt-2 text-xs text-muted-foreground">
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-neuro-cyan" />
          <span>Accuracy</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-neuro-magenta opacity-50 border-dashed" />
          <span>Random Baseline</span>
        </div>
      </div>
    </motion.div>
  );
}
