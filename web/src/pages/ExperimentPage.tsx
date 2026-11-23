import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { fetchRun, fetchTickers, startRun, type TickerOption } from '../api'
import { ChartView } from '../components/ChartView'
import { defaultsFromFields, ParamForm } from '../components/ParamForm'
import { SummaryView } from '../components/SummaryView'
import type { ExperimentMeta, ParamMap, RunStatus } from '../types'

interface ExperimentPageProps {
  experiments: ExperimentMeta[]
  results: Record<string, RunStatus>
  onResult: (experimentId: string, run: RunStatus) => void
}

const STATUS_LABEL: Record<string, string> = {
  queued: 'В очереди',
  preparing_sample: 'Готовим выборку',
  running: 'Считаем',
  done: 'Готово',
  error: 'Ошибка',
}

export function ExperimentPage({ experiments, results, onResult }: ExperimentPageProps) {
  const { experimentId = '' } = useParams()
  const experiment = experiments.find((item) => item.id === experimentId)

  const [params, setParams] = useState<ParamMap>({})
  const [sample, setSample] = useState<ParamMap>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const [liveStatus, setLiveStatus] = useState<string | null>(null)
  const [compareOn, setCompareOn] = useState(false)
  const [compareTicker, setCompareTicker] = useState('')
  const [tickerOptions, setTickerOptions] = useState<TickerOption[]>([])

  useEffect(() => {
    if (!experiment) return
    setParams(defaultsFromFields(experiment.fields))
    setSample(defaultsFromFields(experiment.sample_fields))
    setError(null)
    setActiveRunId(null)
    setLiveStatus(null)
    setCompareOn(false)
    setCompareTicker('')
  }, [experiment])

  useEffect(() => {
    if (experiment?.id !== 'correlation') return
    let cancelled = false
    fetchTickers()
      .then((rows) => {
        if (cancelled) return
        setTickerOptions(rows)
        setCompareTicker((current) => {
          if (current && rows.some((row) => row.ticker === current)) return current
          return rows[0]?.ticker ?? ''
        })
      })
      .catch(() => {
        if (!cancelled) setTickerOptions([])
      })
    return () => {
      cancelled = true
    }
  }, [experiment?.id])

  useEffect(() => {
    if (!activeRunId || !experiment) return
    let cancelled = false
    const timer = window.setInterval(async () => {
      try {
        const run = await fetchRun(activeRunId)
        if (cancelled) return
        setLiveStatus(run.status)
        if (run.status === 'done' || run.status === 'error') {
          onResult(experiment.id, run)
          setBusy(false)
          setActiveRunId(null)
          if (run.status === 'error') setError(run.error ?? 'Ошибка запуска')
        }
      } catch (err) {
        if (cancelled) return
        setBusy(false)
        setActiveRunId(null)
        setError(err instanceof Error ? err.message : String(err))
      }
    }, 1000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [activeRunId, experiment, onResult])

  const stored = experiment ? results[experiment.id] : undefined
  const charts = stored?.status === 'done' ? stored.charts : []
  const summary = stored?.status === 'done' ? stored.summary : null
  const message = stored?.status === 'done' ? stored.message : null

  const statusText = useMemo(() => {
    if (liveStatus) return STATUS_LABEL[liveStatus] ?? liveStatus
    if (busy) return 'Запуск…'
    return null
  }, [busy, liveStatus])

  if (!experiments.length) {
    return <p className="muted">Загрузка каталога экспериментов…</p>
  }

  if (!experiment) {
    return <p className="muted">Эксперимент не найден.</p>
  }

  async function handleRun() {
    if (!experiment) return
    const current = experiment
    const runParams: ParamMap = { ...params }
    if (current.id === 'correlation') {
      runParams.compare = compareOn
      if (compareOn) {
        if (!compareTicker) {
          setError('Выберите акцию для сравнения')
          return
        }
        runParams.ticker = compareTicker
      }
    }
    setError(null)
    setBusy(true)
    setLiveStatus('queued')
    try {
      const response = await startRun({
        experiment: current.id,
        params: runParams,
        sample,
      })
      setActiveRunId(response.run_id)
    } catch (err) {
      setBusy(false)
      setLiveStatus(null)
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div className="experiment-page">
      <header className="page-header">
        <div>
          <h1>{experiment.title}</h1>
          <p className="muted">{experiment.description}</p>
        </div>
        <button type="button" className="primary" disabled={busy} onClick={handleRun}>
          {busy ? 'Считаем…' : 'Запустить'}
        </button>
      </header>

      <section className="panel">
        <h2>Выборка</h2>
        <ParamForm
          fields={experiment.sample_fields}
          values={sample}
          disabled={busy}
          onChange={(name, value) => setSample((prev) => ({ ...prev, [name]: value }))}
        />
      </section>

      <section className="panel">
        <h2>Параметры эксперимента</h2>
        <ParamForm
          fields={experiment.fields}
          values={params}
          disabled={busy}
          onChange={(name, value) => setParams((prev) => ({ ...prev, [name]: value }))}
        />
      </section>

      {experiment.id === 'correlation' && (
        <section className="panel">
          <h2>Режим сравнения</h2>
          <div className="compare-row">
            <label className="param-field checkbox compare-toggle">
              <input
                type="checkbox"
                checked={compareOn}
                disabled={busy}
                onChange={(event) => setCompareOn(event.target.checked)}
              />
              <span>Включить режим сравнения</span>
            </label>
            <label className="param-field">
              <span>Акция</span>
              <select
                value={compareTicker}
                disabled={busy || !compareOn || tickerOptions.length === 0}
                onChange={(event) => setCompareTicker(event.target.value)}
              >
                {tickerOptions.length === 0 && <option value="">Нет акций в выборке</option>}
                {tickerOptions.map((item) => (
                  <option key={item.ticker} value={item.ticker}>
                    {item.ticker} — {item.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </section>
      )}

      {statusText && <div className="banner info">{statusText}</div>}
      {error && <div className="banner error">{error}</div>}

      {(summary || message) && (
        <section className="panel">
          <h2>Сводка</h2>
          <SummaryView summary={summary} message={message} />
        </section>
      )}

      {charts.length > 0 && (
        <section className="charts">
          <h2>Графики</h2>
          {charts.map((chart) => (
            <ChartView key={chart.id} chart={chart} />
          ))}
        </section>
      )}
    </div>
  )
}
