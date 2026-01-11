"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { motion } from "framer-motion";

interface PlasticityCurveProps {
  data: Array<{ shot_level: number; compliance_rate: number }>;
  title?: string;
}

export function PlasticityCurve({ 
  data, 
  title = "Compliance vs. Jailbreak Shots" 
}: PlasticityCurveProps) {
  const formattedData = data.map((d) => ({
    shots: d.shot_level,
    compliance: d.compliance_rate * 100,
    refusal: (1 - d.compliance_rate) * 100,
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
        <AreaChart
          data={formattedData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <defs>
            <linearGradient id="complianceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#FF006E" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#FF006E" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="refusalGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10B981" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis
            dataKey="shots"
            stroke="#525252"
            tick={{ fill: "#a3a3a3", fontSize: 11 }}
            label={{
              value: "N-Shot Examples",
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
              value: "Rate %",
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
            formatter={(value: number, name: string) => [
              `${value.toFixed(1)}%`,
              name === "compliance" ? "Compliance (Bad)" : "Refusal (Good)",
            ]}
            labelFormatter={(label) => `${label}-shot`}
          />
          <Area
            type="monotone"
            dataKey="compliance"
            stroke="#FF006E"
            strokeWidth={2}
            fill="url(#complianceGradient)"
          />
          <Area
            type="monotone"
            dataKey="refusal"
            stroke="#10B981"
            strokeWidth={2}
            fill="url(#refusalGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
      <div className="flex items-center justify-center gap-4 mt-2 text-xs text-muted-foreground">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-neuro-magenta/40" />
          <span>Compliance (Harmful)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-risk-low/40" />
          <span>Refusal (Safe)</span>
        </div>
      </div>
    </motion.div>
  );
}
