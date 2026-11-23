import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { fetchExperiments } from './api'
import { ExperimentPage } from './pages/ExperimentPage'
import type { ExperimentMeta, RunStatus } from './types'
import './App.css'

export default function App() {
  const [experiments, setExperiments] = useState<ExperimentMeta[]>([])
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<Record<string, RunStatus>>({})

  useEffect(() => {
    fetchExperiments()
      .then(setExperiments)
      .catch((err: Error) => setError(err.message))
  }, [])

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <strong>Tinvest</strong>
          <span>эксперименты</span>
        </div>
        <nav>
          {experiments.map((item) => (
            <NavLink
              key={item.id}
              to={`/experiments/${item.id}`}
              className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}
            >
              {item.title}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="content">
        {error && <div className="banner error">{error}</div>}
        <Routes>
          <Route
            path="/"
            element={
              experiments[0] ? (
                <Navigate to={`/experiments/${experiments[0].id}`} replace />
              ) : (
                <p className="muted">Загрузка каталога экспериментов…</p>
              )
            }
          />
          <Route
            path="/experiments/:experimentId"
            element={
              <ExperimentPage
                experiments={experiments}
                results={results}
                onResult={(experimentId, run) =>
                  setResults((prev) => ({ ...prev, [experimentId]: run }))
                }
              />
            }
          />
        </Routes>
      </main>
    </div>
  )
}
