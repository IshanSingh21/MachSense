import {
  BatchPredictionResponse,
  HealthResponse,
  PredictionResponse,
  ReadinessResponse,
  TelemetryItem,
} from "@/types";

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    public status: number,
    public message: string,
    public details: string[] = []
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// Physical boundary constraints matching backend domain validator
export const SENSOR_LIMITS = {
  air_temperature_k: { min: 295.0, max: 305.0, unit: "K", label: "Air Temperature" },
  process_temperature_k: { min: 305.0, max: 315.0, unit: "K", label: "Process Temperature" },
  rotational_speed_rpm: { min: 1100.0, max: 2900.0, unit: "RPM", label: "Rotational Speed" },
  torque_nm: { min: 3.0, max: 80.0, unit: "Nm", label: "Mechanical Torque" },
  tool_wear_min: { min: 0.0, max: 260.0, unit: "min", label: "Tool Wear" },
};

export function validateTelemetryInput(t: TelemetryItem): string[] {
  const errors: string[] = [];

  if (t.air_temperature_k < SENSOR_LIMITS.air_temperature_k.min || t.air_temperature_k > SENSOR_LIMITS.air_temperature_k.max) {
    errors.push(`Air temperature must be between ${SENSOR_LIMITS.air_temperature_k.min} and ${SENSOR_LIMITS.air_temperature_k.max} K.`);
  }
  if (t.process_temperature_k < SENSOR_LIMITS.process_temperature_k.min || t.process_temperature_k > SENSOR_LIMITS.process_temperature_k.max) {
    errors.push(`Process temperature must be between ${SENSOR_LIMITS.process_temperature_k.min} and ${SENSOR_LIMITS.process_temperature_k.max} K.`);
  }
  if (t.process_temperature_k <= t.air_temperature_k) {
    errors.push("Process temperature must strictly exceed ambient air temperature (thermodynamic invariant).");
  }
  if (t.rotational_speed_rpm < SENSOR_LIMITS.rotational_speed_rpm.min || t.rotational_speed_rpm > SENSOR_LIMITS.rotational_speed_rpm.max) {
    errors.push(`Rotational speed must be between ${SENSOR_LIMITS.rotational_speed_rpm.min} and ${SENSOR_LIMITS.rotational_speed_rpm.max} RPM.`);
  }
  if (t.torque_nm < SENSOR_LIMITS.torque_nm.min || t.torque_nm > SENSOR_LIMITS.torque_nm.max) {
    errors.push(`Torque must be between ${SENSOR_LIMITS.torque_nm.min} and ${SENSOR_LIMITS.torque_nm.max} Nm.`);
  }
  if (t.tool_wear_min < SENSOR_LIMITS.tool_wear_min.min || t.tool_wear_min > SENSOR_LIMITS.tool_wear_min.max) {
    errors.push(`Tool wear must be between ${SENSOR_LIMITS.tool_wear_min.min} and ${SENSOR_LIMITS.tool_wear_min.max} minutes.`);
  }

  return errors;
}

async function request<T>(
  path: string,
  options: RequestInit = {},
  retries = 1
): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        ...options.headers,
      },
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      let errorDetails: string[] = [];
      let errorMessage = `HTTP ${res.status}: ${res.statusText}`;

      try {
        const errorJson = await res.json();
        errorMessage = errorJson.message || errorMessage;
        errorDetails = errorJson.details || [];
      } catch {
        // Non-JSON response
      }

      throw new ApiError(res.status, errorMessage, errorDetails);
    }

    return (await res.json()) as T;
  } catch (err: unknown) {
    clearTimeout(timeoutId);

    if (retries > 0 && !(err instanceof ApiError && err.status >= 400 && err.status < 500)) {
      // Retry once on network failure or 5xx server error
      await new Promise((r) => setTimeout(r, 600));
      return request<T>(path, options, retries - 1);
    }

    if (err instanceof ApiError) {
      throw err;
    }

    const message =
      err instanceof Error
        ? err.name === "AbortError"
          ? "Request timed out (10s limit). Verify the FastAPI backend is responsive."
          : err.message
        : "Failed to communicate with MachSense backend.";
    throw new ApiError(0, message);
  }
}

export const api = {
  getBaseUrl(): string {
    return API_BASE_URL;
  },

  async getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>("/health");
  },

  async getReadiness(): Promise<ReadinessResponse> {
    return request<ReadinessResponse>("/health/ready");
  },

  async predictSingle(
    telemetry: TelemetryItem,
    explain: boolean = true,
    threshold?: number
  ): Promise<PredictionResponse> {
    // Validate inputs locally first
    const validationErrors = validateTelemetryInput(telemetry);
    if (validationErrors.length > 0) {
      throw new ApiError(422, "Input telemetry violates physical boundaries.", validationErrors);
    }

    return request<PredictionResponse>("/api/v1/predict", {
      method: "POST",
      body: JSON.stringify({
        telemetry,
        explain,
        threshold: threshold !== undefined ? threshold : null,
      }),
    });
  },

  async predictBatch(
    items: TelemetryItem[],
    explain: boolean = false,
    threshold?: number
  ): Promise<BatchPredictionResponse> {
    if (items.length === 0) {
      throw new ApiError(422, "Batch payload must contain at least 1 record.");
    }
    if (items.length > 5000) {
      throw new ApiError(422, "Batch size exceeds maximum limit of 5,000 records.");
    }

    return request<BatchPredictionResponse>("/api/v1/predict/batch", {
      method: "POST",
      body: JSON.stringify({
        items,
        explain,
        threshold: threshold !== undefined ? threshold : null,
      }),
    });
  },
};
