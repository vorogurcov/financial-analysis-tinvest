import type { FieldSpec, ParamMap } from '../types'

interface ParamFormProps {
  fields: FieldSpec[]
  values: ParamMap
  onChange: (name: string, value: string | number | boolean) => void
  disabled?: boolean
}

export function ParamForm({ fields, values, onChange, disabled }: ParamFormProps) {
  if (!fields.length) {
    return <p className="muted">У этого эксперимента нет дополнительных параметров.</p>
  }

  return (
    <div className="param-grid">
      {fields.map((field) => {
        const value = values[field.name] ?? field.default
        if (field.type === 'boolean') {
          return (
            <label key={field.name} className="param-field checkbox">
              <input
                type="checkbox"
                checked={Boolean(value)}
                disabled={disabled}
                onChange={(event) => onChange(field.name, event.target.checked)}
              />
              <span>{field.label}</span>
            </label>
          )
        }

        return (
          <label key={field.name} className="param-field">
            <span>{field.label}</span>
            <input
              type="number"
              value={Number(value)}
              disabled={disabled}
              min={field.min ?? undefined}
              max={field.max ?? undefined}
              step={field.step ?? (field.type === 'integer' ? 1 : 'any')}
              onChange={(event) => {
                const raw = event.target.value
                if (raw === '') return
                const parsed = field.type === 'integer' ? Number.parseInt(raw, 10) : Number.parseFloat(raw)
                if (!Number.isNaN(parsed)) onChange(field.name, parsed)
              }}
            />
          </label>
        )
      })}
    </div>
  )
}

export function defaultsFromFields(fields: FieldSpec[]): ParamMap {
  const out: ParamMap = {}
  for (const field of fields) {
    out[field.name] = field.default
  }
  return out
}
