import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import type { ChartSpec } from '../types'

function buildOption(chart: ChartSpec): EChartsOption {
  switch (chart.kind) {
    case 'line':
      return lineOption(chart)
    case 'dual_line':
      return dualLineOption(chart)
    case 'histogram':
      return histogramOption(chart)
    case 'scatter':
      return scatterOption(chart)
    case 'boxplot':
      return boxplotOption(chart)
    case 'errorbar':
      return errorbarOption(chart)
    case 'heatmap':
      return heatmapOption(chart)
    case 'bar':
      return barOption(chart)
    case 'grouped_bar':
      return groupedBarOption(chart)
    case 'qq_grid':
      return { title: { text: chart.title } }
    default:
      return { title: { text: chart.title } }
  }
}

function lineOption(chart: ChartSpec): EChartsOption {
  const hasOwnX = chart.series?.some((s) => s.x && s.x.length)
  const series = (chart.series ?? []).flatMap((item) => {
    const out: EChartsOption['series'] = []
    if (item.band_low && item.band_high) {
      out.push({
        name: `${item.name} band`,
        type: 'line',
        data: item.band_low,
        lineStyle: { opacity: 0 },
        stack: `${item.name}-band`,
        symbol: 'none',
        silent: true,
        areaStyle: { color: 'transparent' },
      })
      out.push({
        name: `${item.name} band`,
        type: 'line',
        data: item.band_high.map((high, index) => {
          const low = item.band_low?.[index]
          if (high == null || low == null) return null
          return high - low
        }),
        lineStyle: { opacity: 0 },
        stack: `${item.name}-band`,
        symbol: 'none',
        areaStyle: { color: 'rgba(70, 130, 180, 0.18)' },
        silent: true,
      })
    }
    out.push({
      name: item.name,
      type: 'line',
      showSymbol: (item.values?.length ?? 0) < 40,
      data: hasOwnX && item.x
        ? item.x.map((x, i) => [x, item.values[i] ?? null])
        : item.values,
    })
    return out
  })

  return {
    title: { text: chart.title, left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { top: 28 },
    grid: { left: 56, right: 28, top: 72, bottom: 56 },
    xAxis: hasOwnX
      ? { type: 'value', name: chart.x_label ?? undefined, scale: true }
      : {
          type: 'category',
          data: chart.x ?? undefined,
          name: chart.x_label ?? undefined,
          axisLabel: { hideOverlap: true },
        },
    yAxis: { type: 'value', name: chart.y_label ?? undefined, scale: true },
    dataZoom: [{ type: 'inside' }, { type: 'slider' }],
    series,
  }
}

function dualLineOption(chart: ChartSpec): EChartsOption {
  return {
    title: { text: chart.title, left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { top: 28 },
    grid: { left: 56, right: 56, top: 72, bottom: 56 },
    xAxis: {
      type: 'category',
      data: chart.x ?? undefined,
      name: chart.x_label ?? undefined,
      axisLabel: { hideOverlap: true },
    },
    yAxis: [
      { type: 'value', name: chart.y_label ?? undefined, scale: true },
      { type: 'value', name: chart.y_label_right ?? undefined, scale: true },
    ],
    dataZoom: [{ type: 'inside' }, { type: 'slider' }],
    series: (chart.series ?? []).map((item) => ({
      name: item.name,
      type: 'line',
      showSymbol: false,
      yAxisIndex: item.axis === 'right' ? 1 : 0,
      data: item.values,
    })),
  }
}

function histogramOption(chart: ChartSpec): EChartsOption {
  const bins = chart.bins ?? []
  const centers = bins.map((b) => (b.x0 + b.x1) / 2)
  const densities = bins.map((b) => b.density)
  const ref = chart.reference_line ?? []
  const width = bins.length ? bins[0].x1 - bins[0].x0 : 0.1
  return {
    title: { text: chart.title, left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { top: 28 },
    grid: { left: 56, right: 28, top: 72, bottom: 48 },
    xAxis: { type: 'value', name: chart.x_label ?? undefined, scale: true },
    yAxis: { type: 'value', name: chart.y_label ?? undefined },
    series: [
      {
        name: 'Данные',
        type: 'custom',
        renderItem: (_params, api) => {
          const xCenter = api.value(0) as number
          const yValue = api.value(1) as number
          const top = api.coord([xCenter, yValue])
          const bottom = api.coord([xCenter, 0])
          const left = api.coord([xCenter - width / 2, 0])
          const right = api.coord([xCenter + width / 2, 0])
          return {
            type: 'rect',
            shape: {
              x: left[0],
              y: top[1],
              width: Math.max(right[0] - left[0], 1),
              height: Math.max(bottom[1] - top[1], 0),
            },
            style: { fill: 'rgba(76, 120, 168, 0.7)' },
          }
        },
        data: centers.map((x, i) => [x, densities[i]]),
        z: 1,
      },
      {
        name: 'N(0, 1)',
        type: 'line',
        showSymbol: false,
        data: ref.map((p) => [p.x, p.y]),
        lineStyle: { color: '#111', width: 2 },
        z: 2,
      },
    ],
  }
}

function scatterOption(chart: ChartSpec): EChartsOption {
  const ref = chart.reference_line ?? []
  const points = (chart.series ?? []).map((item) => ({
    name: item.name,
    type: 'scatter' as const,
    symbolSize: 8,
    data: (item.x ?? []).map((x, i) => [x, item.values[i] ?? null]),
  }))
  const line =
    ref.length >= 2
      ? [
          {
            name: 'Регрессия',
            type: 'line' as const,
            showSymbol: false,
            data: ref.map((point) => [point.x, point.y]),
            lineStyle: { color: '#111', width: 2 },
            z: 2,
          },
        ]
      : []
  return {
    title: { text: chart.title, left: 'center' },
    tooltip: { trigger: 'item' },
    legend: { top: 28, type: 'scroll' },
    grid: { left: 56, right: 28, top: 72, bottom: 48 },
    xAxis: { type: 'value', name: chart.x_label ?? undefined, scale: true },
    yAxis: { type: 'value', name: chart.y_label ?? undefined, scale: true },
    series: [...points, ...line],
  }
}

function boxplotOption(chart: ChartSpec): EChartsOption {
  const boxes = chart.boxes ?? []
  return {
    title: { text: chart.title, left: 'center' },
    tooltip: { trigger: 'item' },
    grid: { left: 56, right: 28, top: 64, bottom: 64 },
    xAxis: {
      type: 'category',
      data: boxes.map((b) => b.label),
      name: chart.x_label ?? undefined,
      axisLabel: { rotate: boxes.length > 6 ? 30 : 0 },
    },
    yAxis: { type: 'value', name: chart.y_label ?? undefined, scale: true },
    series: [
      {
        name: chart.y_label ?? 'boxplot',
        type: 'boxplot',
        data: boxes.map((b) => [b.min, b.q1, b.median, b.q3, b.max]),
      },
      {
        name: 'Выбросы',
        type: 'scatter',
        data: boxes.flatMap((b, index) => b.outliers.map((value) => [index, value])),
      },
    ],
  }
}

function errorbarOption(chart: ChartSpec): EChartsOption {
  const series = chart.series?.[0]
  const xs = chart.x ?? []
  const values = series?.values ?? []
  const low = series?.yerr_low ?? []
  const high = series?.yerr_high ?? []
  return {
    title: { text: chart.title, left: 'center' },
    tooltip: { trigger: 'axis' },
    grid: { left: 56, right: 28, top: 64, bottom: 48 },
    xAxis: {
      type: 'category',
      data: xs.map(String),
      name: chart.x_label ?? undefined,
    },
    yAxis: { type: 'value', name: chart.y_label ?? undefined, scale: true },
    series: [
      {
        name: series?.name ?? 'series',
        type: 'scatter',
        data: values,
        symbolSize: 10,
        itemStyle: { color: '#4C78A8' },
      },
      {
        name: '95% интервал',
        type: 'custom',
        renderItem: (_params, api) => {
          const xValue = api.value(0)
          const highValue = api.value(1)
          const lowValue = api.value(2)
          const pointHigh = api.coord([xValue, highValue])
          const pointLow = api.coord([xValue, lowValue])
          const halfWidth = 6
          return {
            type: 'group',
            children: [
              {
                type: 'line',
                shape: { x1: pointHigh[0], y1: pointHigh[1], x2: pointLow[0], y2: pointLow[1] },
                style: { stroke: '#4C78A8', lineWidth: 1.5 },
              },
              {
                type: 'line',
                shape: {
                  x1: pointHigh[0] - halfWidth,
                  y1: pointHigh[1],
                  x2: pointHigh[0] + halfWidth,
                  y2: pointHigh[1],
                },
                style: { stroke: '#4C78A8', lineWidth: 1.5 },
              },
              {
                type: 'line',
                shape: {
                  x1: pointLow[0] - halfWidth,
                  y1: pointLow[1],
                  x2: pointLow[0] + halfWidth,
                  y2: pointLow[1],
                },
                style: { stroke: '#4C78A8', lineWidth: 1.5 },
              },
            ],
          }
        },
        data: values.map((value, index) => {
          if (value == null) return null
          const errLow = low[index] ?? 0
          const errHigh = high[index] ?? 0
          return [index, value + errHigh, value - errLow]
        }),
        z: 0,
      },
    ],
  }
}

function heatmapOption(chart: ChartSpec): EChartsOption {
  const matrix = chart.matrix ?? []
  const xLabels = chart.x_labels ?? []
  const yLabels = chart.y_labels ?? []
  const data: Array<[number, number, number | null]> = []
  matrix.forEach((row, y) => {
    row.forEach((value, x) => {
      data.push([x, y, value])
    })
  })
  return {
    title: { text: chart.title, left: 'center' },
    tooltip: { position: 'top' },
    grid: { left: 120, right: 48, top: 64, bottom: 120 },
    xAxis: {
      type: 'category',
      data: xLabels,
      axisLabel: { rotate: 90, fontSize: 9 },
      splitArea: { show: true },
    },
    yAxis: {
      type: 'category',
      data: yLabels,
      axisLabel: { fontSize: 9 },
      splitArea: { show: true },
    },
    visualMap: {
      min: -1,
      max: 1,
      calculable: true,
      orient: 'vertical',
      right: 0,
      top: 'center',
      inRange: { color: ['#2166ac', '#f7f7f7', '#b2182b'] },
    },
    series: [
      {
        name: 'Pearson',
        type: 'heatmap',
        data,
        emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,0.3)' } },
      },
    ],
  }
}

function barOption(chart: ChartSpec): EChartsOption {
  if (chart.horizontal) {
    return {
      title: { text: chart.title, left: 'center' },
      tooltip: { trigger: 'axis' },
      grid: { left: 120, right: 28, top: 64, bottom: 48 },
      xAxis: { type: 'value', name: chart.y_label ?? undefined, scale: true },
      yAxis: {
        type: 'category',
        data: chart.categories ?? undefined,
      },
      series: [
        {
          type: 'bar',
          data: chart.values ?? [],
          itemStyle: { color: '#4C78A8' },
        },
      ],
    }
  }
  return {
    title: { text: chart.title, left: 'center' },
    tooltip: { trigger: 'axis' },
    grid: { left: 56, right: 28, top: 64, bottom: 48 },
    xAxis: {
      type: 'category',
      data: chart.categories ?? undefined,
      axisLabel: { hideOverlap: true },
    },
    yAxis: { type: 'value', name: chart.y_label ?? undefined, scale: true },
    series: [
      {
        type: 'bar',
        data: chart.values ?? [],
        itemStyle: { color: '#4C78A8' },
      },
    ],
  }
}

function groupedBarOption(chart: ChartSpec): EChartsOption {
  return {
    title: { text: chart.title, left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { top: 28 },
    grid: { left: 56, right: 28, top: 72, bottom: 48 },
    xAxis: { type: 'category', data: chart.categories ?? undefined },
    yAxis: { type: 'value', name: chart.y_label ?? undefined, max: 1.05 },
    series: (chart.groups ?? []).map((group) => ({
      name: group.name,
      type: 'bar',
      data: group.values,
    })),
  }
}

export function ChartView({ chart }: { chart: ChartSpec }) {
  if (chart.kind === 'qq_grid') {
    const panels = chart.panels ?? []
    return (
      <section className="chart-card">
        <h3>{chart.title}</h3>
        <div className="qq-grid">
          {panels.map((panel) => (
            <ReactECharts
              key={panel.id}
              option={scatterOption(panel)}
              style={{ height: 220, width: '100%' }}
              notMerge
              lazyUpdate
            />
          ))}
        </div>
      </section>
    )
  }

  return (
    <section className="chart-card">
      <ReactECharts
        option={buildOption(chart)}
        style={{ height: chart.kind === 'heatmap' ? 560 : 420, width: '100%' }}
        notMerge
        lazyUpdate
      />
    </section>
  )
}
