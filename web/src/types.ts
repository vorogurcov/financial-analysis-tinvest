export type FieldType = 'number' | 'integer' | 'boolean' | 'string'

export interface FieldSpec {
  name: string
  label: string
  type: FieldType
  default: string | number | boolean
  min?: number | null
  max?: number | null
  step?: number | null
  required?: boolean
}

export interface ExperimentMeta {
  id: string
  title: string
  description: string
  needs_sample: boolean
  fields: FieldSpec[]
  sample_fields: FieldSpec[]
}

export interface ChartSeries {
  name: string
  values: Array<number | null>
  axis?: 'left' | 'right' | null
  x?: Array<number | null> | null
  yerr_low?: Array<number | null> | null
  yerr_high?: Array<number | null> | null
  band_low?: Array<number | null> | null
  band_high?: Array<number | null> | null
}

export interface HistogramBin {
  x0: number
  x1: number
  density: number
}

export interface BoxStats {
  label: string
  min: number
  q1: number
  median: number
  q3: number
  max: number
  outliers: number[]
}

export interface ChartSpec {
  id: string
  title: string
  kind:
    | 'line'
    | 'histogram'
    | 'scatter'
    | 'boxplot'
    | 'errorbar'
    | 'heatmap'
    | 'dual_line'
    | 'bar'
    | 'grouped_bar'
    | 'qq_grid'
  x?: Array<string | number> | null
  series?: ChartSeries[]
  x_label?: string | null
  y_label?: string | null
  y_label_right?: string | null
  bins?: HistogramBin[] | null
  reference_line?: Array<{ x: number; y: number }> | null
  boxes?: BoxStats[] | null
  categories?: string[] | null
  values?: Array<number | null> | null
  groups?: Array<{ name: string; values: Array<number | null> }> | null
  matrix?: Array<Array<number | null>> | null
  x_labels?: string[] | null
  y_labels?: string[] | null
  panels?: ChartSpec[] | null
  horizontal?: boolean
}

export interface RunStatus {
  id: string
  experiment: string
  status: 'queued' | 'preparing_sample' | 'running' | 'done' | 'error'
  created_at: string
  updated_at: string
  error: string | null
  summary: Record<string, unknown> | Array<Record<string, unknown>> | null
  charts: ChartSpec[]
  message: string | null
}

export type ParamMap = Record<string, string | number | boolean>
