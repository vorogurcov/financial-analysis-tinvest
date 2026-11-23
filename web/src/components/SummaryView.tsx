interface SummaryViewProps {
  summary: Record<string, unknown> | Array<Record<string, unknown>> | null
  message?: string | null
}

const NESTED_TITLES: Record<string, string> = {
  compare: 'Сравнение',
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
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

function SummaryGrid({ data }: { data: Record<string, unknown> }) {
  return (
    <dl className="summary-grid">
      {Object.entries(data).map(([key, value]) => (
        <div key={key} className="summary-item">
          <dt>{key}</dt>
          <dd>{formatValue(value)}</dd>
        </div>
      ))}
    </dl>
  )
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

  const entries = Object.entries(summary)
  const flat = Object.fromEntries(entries.filter(([, value]) => !isRecord(value)))
  const nested = entries.filter((entry): entry is [string, Record<string, unknown>] => isRecord(entry[1]))

  return (
    <>
      {Object.keys(flat).length > 0 && <SummaryGrid data={flat} />}
      {nested.map(([key, value]) => (
        <div key={key} className="summary-nested">
          <h3>{NESTED_TITLES[key] ?? key}</h3>
          <SummaryGrid data={value} />
        </div>
      ))}
    </>
  )
}
