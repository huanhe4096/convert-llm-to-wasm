<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const modelId = ref('mixedbread-ai/mxbai-embed-xsmall-v1')

const maxSentences = 100000
const umapFitSampleSize = 8192
const umapTransformBatchSize = 16384
const randomCount = ref(500)
const batchSize = ref(100)
const targetDim = ref(128)
const sentenceInput = ref('')
const activeSentences = ref([])
const points = ref([])
const stage = ref('idle')
const statusText = ref('Ready')
const progress = ref(0)
const elapsedMs = ref(0)
const embeddingDim = ref(0)
const usedDim = ref(0)
const isRunning = ref(false)

let worker = null
let currentRunId = 0

const parsedSentences = computed(() =>
  sentenceInput.value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean),
)

const pointView = computed(() => {
  if (!points.value.length) return []

  const plotWidth = 1000
  const plotHeight = 700
  const padding = 50

  const xs = points.value.map((p) => p.x)
  const ys = points.value.map((p) => p.y)

  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)

  const xSpan = maxX - minX || 1
  const ySpan = maxY - minY || 1

  return points.value.map((point, index) => ({
    index,
    label: activeSentences.value[point.sentenceIndex] ?? `Sentence #${point.sentenceIndex + 1}`,
    x: padding + ((point.x - minX) / xSpan) * (plotWidth - padding * 2),
    y: plotHeight - padding - ((point.y - minY) / ySpan) * (plotHeight - padding * 2),
  }))
})

function randomPick(list) {
  return list[Math.floor(Math.random() * list.length)]
}

function createRandomSentence() {
  const subjects = ['The analyst', 'A researcher', 'The designer', 'An engineer', 'The student', 'A manager']
  const verbs = ['compares', 'evaluates', 'summarizes', 'visualizes', 'clusters', 'embeds']
  const objects = ['customer feedback', 'medical abstracts', 'news headlines', 'product reviews', 'incident reports', 'research notes']
  const places = ['in a browser demo', 'for a weekly report', 'during a sprint review', 'inside a data notebook', 'for rapid prototyping']
  const modifiers = ['with careful tuning', 'under strict latency limits', 'using lightweight models', 'before lunch', 'with a tiny dataset']

  return `${randomPick(subjects)} ${randomPick(verbs)} ${randomPick(objects)} ${randomPick(places)} ${randomPick(modifiers)}.`
}

function sampleIndices(total, sampleSize) {
  const indices = Array.from({ length: total }, (_, i) => i)
  for (let i = 0; i < sampleSize; i += 1) {
    const j = i + Math.floor(Math.random() * (total - i))
    const tmp = indices[i]
    indices[i] = indices[j]
    indices[j] = tmp
  }
  return indices.slice(0, sampleSize)
}

function generateRandomSentences() {
  const n = Math.max(2, Math.min(maxSentences, Number(randomCount.value) || 0))
  sentenceInput.value = Array.from({ length: n }, () => createRandomSentence()).join('\n')
  activeSentences.value = []
  points.value = []
  stage.value = 'idle'
  statusText.value = `Generated ${n} random sentences.`
  progress.value = 0
  elapsedMs.value = 0
  embeddingDim.value = 0
  usedDim.value = 0
}

function ensureWorker() {
  if (worker) return

  worker = new Worker(new URL('./workers/embedding.worker.js', import.meta.url), {
    type: 'module',
  })

  worker.onmessage = (event) => {
    const message = event.data
    if (message.runId !== currentRunId) return

    if (message.type === 'progress') {
      if (message.stage) stage.value = message.stage
      if (message.statusText) statusText.value = message.statusText
      if (typeof message.progress === 'number') progress.value = message.progress
      if (typeof message.embeddingDim === 'number') embeddingDim.value = message.embeddingDim
      if (typeof message.usedDim === 'number') usedDim.value = message.usedDim
      return
    }

    if (message.type === 'points-reset') {
      points.value = message.points
      return
    }

    if (message.type === 'points-append') {
      points.value.push(...message.points)
      return
    }

    if (message.type === 'done') {
      progress.value = 1
      stage.value = 'done'
      statusText.value = `Done. Embedded ${message.totalCount} sentences.`
      elapsedMs.value = message.elapsedMs
      isRunning.value = false
      return
    }

    if (message.type === 'error') {
      stage.value = 'error'
      statusText.value = `Failed: ${message.message}`
      isRunning.value = false
    }
  }

  worker.onerror = (event) => {
    stage.value = 'error'
    statusText.value = `Worker error: ${event.message}`
    isRunning.value = false
  }
}

async function runEmbedding() {
  const sentences = parsedSentences.value

  if (!sentences.length) {
    statusText.value = 'Please provide at least one sentence.'
    return
  }

  if (!navigator.gpu) {
    stage.value = 'error'
    statusText.value = 'WebGPU is not available in this browser.'
    return
  }

  if (typeof Worker === 'undefined') {
    stage.value = 'error'
    statusText.value = 'Web Worker is not supported in this browser.'
    return
  }

  isRunning.value = true
  stage.value = 'loading-model'
  progress.value = 0
  elapsedMs.value = 0
  embeddingDim.value = 0
  usedDim.value = 0
  statusText.value = `Loading model ${modelId.value}...`
  points.value = []
  activeSentences.value = Array.from(sentences, (s) => String(s))
  const sentencePayload = Array.from(activeSentences.value, (s) => String(s))

  ensureWorker()
  currentRunId += 1

  worker.postMessage({
    type: 'run',
    runId: currentRunId,
    modelId: modelId.value,
    dtype: 'q4',
    sentences: sentencePayload,
    embeddingBatchSize: Math.max(1, Math.min(1000, Number(batchSize.value) || 1)),
    targetDim: Number(targetDim.value) || 128,
    umapFitSampleSize,
    umapTransformBatchSize,
  })
}

onMounted(() => {
  generateRandomSentences()
})

onBeforeUnmount(() => {
  if (worker) {
    worker.terminate()
    worker = null
  }
})
</script>

<template>
  <main class="layout">
    <section class="left-column">
      <header class="panel hero">
        <p class="eyebrow">WebGPU + Transformers.js v3</p>
        <h1>Online Text Embedding Demo</h1>
        <p class="model">Model: <code>{{ modelId }}</code></p>
      </header>

      <section class="panel controls">
        <div class="row">
          <label for="model-id">Transformers.js model</label>
          <input id="model-id" v-model="modelId" :disabled="isRunning" />
        </div>
        <div class="row">
          <label for="count">Random sentence count</label>
          <input id="count" v-model.number="randomCount" type="number" min="2" :max="maxSentences" />
        </div>
        <div class="row">
          <label for="batch-size">Embedding batch size</label>
          <input id="batch-size" v-model.number="batchSize" type="number" min="1" max="1000" />
        </div>
        <div class="row">
          <label for="target-dim">Used embedding dim</label>
          <select id="target-dim" v-model.number="targetDim" :disabled="isRunning">
            <option :value="384">384</option>
            <option :value="256">256</option>
            <option :value="128">128</option>
          </select>
        </div>
        <div class="buttons">
          <button class="secondary" :disabled="isRunning" @click="generateRandomSentences">Generate Samples</button>
          <button class="primary" :disabled="isRunning || !parsedSentences.length" @click="runEmbedding">
            {{ isRunning ? 'Running...' : 'Start Embedding' }}
          </button>
        </div>
      </section>

      <section class="panel input-panel">
        <div class="panel-title">Input Sentences (one per line)</div>
        <textarea v-model="sentenceInput" :disabled="isRunning" spellcheck="false"></textarea>
        <div class="meta">Total sentences: {{ parsedSentences.length }}</div>
      </section>

      <section class="panel progress-panel">
        <div class="panel-title">Progress</div>
        <div class="status">{{ statusText }}</div>
        <div class="progress-track">
          <div class="progress-bar" :style="{ width: `${Math.round(progress * 100)}%` }"></div>
        </div>
        <div class="stats">
          <span>Stage: {{ stage }}</span>
          <span>Progress: {{ Math.round(progress * 100) }}%</span>
          <span>Embedding dim: {{ embeddingDim || '-' }}</span>
          <span>Used dim: {{ usedDim || '-' }}</span>
          <span>Elapsed: {{ elapsedMs ? `${elapsedMs.toFixed(1)} ms` : '-' }}</span>
        </div>
      </section>
    </section>

    <section class="right-column panel plot-panel">
      <div class="panel-title">UMAP Point Cloud (2D)</div>
      <p class="plot-note">UMAP strategy: fit on random 8192 points, then transform the rest in 16384-sized batches.</p>

      <div v-if="pointView.length" class="plot-frame">
        <svg viewBox="0 0 1000 700" role="img" aria-label="UMAP embedding scatter plot">
          <g v-for="point in pointView" :key="point.index">
            <circle :cx="point.x" :cy="point.y" r="5.5" />
            <title>{{ point.label }}</title>
          </g>
        </svg>
      </div>
      <div v-else class="plot-empty">Run embedding to render the UMAP visualization.</div>
    </section>
  </main>
</template>
