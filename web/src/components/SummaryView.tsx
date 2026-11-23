interface SummaryViewProps {
  summary: Record<string, unknown> | Array<Record<string, unknown>> | null
  message?: string | null
}

function formatValue(value: unknown): string {
  if (value == null) return '—'
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value)
    return Number.isInteger(value) ? String(value) : value.toFixed(4)
  }
  if (typeof value === 'boolean') return value ? 'да' : 'нет'
  return String(value)
}

export function SummaryView({ summary, message }: SummaryViewProps) {
  if (message) {
    return <div className="banner">{message}</div>
  }
  if (!summary) return null

  if (Array.isArray(summary)) {
    return (
      <div className="summary-table-wrap">
        <table className="summary-table">
          <thead>
            <tr>
              {Object.keys(summary[0] ?? {}).map((key) => (
                <th key={key}>{key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {summary.map((row, index) => (
              <tr key={index}>
                {Object.values(row).map((value, cell) => (
                  <td key={cell}>{formatValue(value)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  return (
    <dl className="summary-grid">
      {Object.entries(summary).map(([key, value]) => (
        <div key={key} className="summary-item">
          <dt>{key}</dt>
          <dd>{formatValue(value)}</dd>
        </div>
      ))}
    </dl>
  )
}
