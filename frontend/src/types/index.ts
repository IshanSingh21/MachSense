export type MachineType = "L" | "M" | "H";

export type MachineStatus = "HEALTHY" | "WARNING" | "CRITICAL";

export type RiskLevel = "NOMINAL" | "MODERATE_WARNING" | "ELEVATED" | "CRITICAL";

export interface TelemetryItem {
  type: MachineType;
  air_temperature_k: number;
  process_temperature_k: number;
  rotational_speed_rpm: number;
  torque_nm: number;
  tool_wear_min: number;
  udi?: number;
  product_id?: string;
}

export interface FeatureAttribution {
  feature_name: string;
  display_name?: string;
  attribution_value: number;
  percentage?: number;
  direction?: "increases_risk" | "decreases_risk";
  feature_value?: number;
  unit?: string;
}

export interface PredictionResponse {
  status: "SUCCESS" | "VALIDATION_ERROR" | "PROCESSING_ERROR";
  failure_probability: number;
  predicted_class: number;
  predicted_label: "HEALTHY" | "FAILURE_IMMINENT";
  risk_level: RiskLevel;
  threshold_used: number;
  model_version: string;
  top_risk_escalators: FeatureAttribution[];
  top_stabilizers: FeatureAttribution[];
  operator_summary?: string;
  non_causal_disclaimer?: string;
  latency_ms: number;
}

export interface BatchPredictionItemResult {
  udi?: number;
  product_id?: string;
  failure_probability: number;
  predicted_class: number;
  predicted_label: string;
  risk_level: string;
  threshold_used: number;
  top_risk_escalators?: FeatureAttribution[];
}

export interface BatchPredictionResponse {
  status: string;
  total_records: number;
  successful_predictions_count: number;
  failure_count: number;
  mean_failure_probability: number;
  predictions: BatchPredictionItemResult[];
  total_latency_ms: number;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  environment: string;
}

export interface ReadinessResponse {
  status: string;
  model_loaded: boolean;
  model_name?: string;
  model_version?: string;
  optimal_threshold?: number;
  explainer_ready: boolean;
}

export interface Machine {
  id: string;
  name: string;
  type: MachineType;
  location: string;
  status: MachineStatus;
  failure_probability: number;
  last_prediction_time: string;
  telemetry: TelemetryItem;
  derived: {
    power_w: number;
    temp_diff_k: number;
    overstrain_index: number;
    tool_wear_rate: number;
  };
  warning_message?: string;
}

export interface Alert {
  id: string;
  machine_id: string;
  machine_name: string;
  timestamp: string;
  issue: string;
  failure_type: "HDF" | "PWF" | "OSF" | "TWF" | "RNF" | "GENERAL";
  severity: "CRITICAL" | "WARNING" | "INFO";
  recommended_action: string;
  acknowledged: boolean;
  telemetry_snapshot?: TelemetryItem;
}

export interface TelemetryPreset {
  id: string;
  name: string;
  short_code: "NOMINAL" | "HDF" | "PWF" | "OSF" | "TWF";
  description: string;
  physical_cause: string;
  telemetry: TelemetryItem;
  expected_outcome: string;
}
