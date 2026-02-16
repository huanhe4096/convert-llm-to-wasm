import { pipeline, env } from '@huggingface/transformers'
import { UMAP } from 'umap-js'

env.allowLocalModels = false
env.useBrowserCache = true

let extractorPromise = null
let extractorModelId = null
let activeJobId = 0

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

async function getExtractor(modelId, dtype, progressCallback) {
  if (!extractorPromise || extractorModelId !== modelId) {
    extractorModelId = modelId
    extractorPromise = pipeline('feature-extraction', modelId, {
      device: 'webgpu',
      dtype,
      progress_callback: progressCallback,
    })
  }
  return extractorPromise
}

function post(runId, payload) {
  self.postMessage({ runId, ...payload })
}

function assertActive(jobId) {
  if (jobId !== activeJobId) {
    throw new Error('__CANCELLED__')
  }
}

function sleepFrame() {
  return new Promise((resolve) => setTimeout(resolve, 0))
}

async function runJob(message, jobId) {
  const {
    runId,
    modelId,
    dtype,
    sentences,
    embeddingBatchSize,
    targetDim,
    umapFitSampleSize,
    umapTransformBatchSize,
  } = message

  const started = performance.now()

  try {
    if (!sentences?.length) {
      throw new Error('No sentences provided.')
    }

    post(runId, {
      type: 'progress',
      stage: 'loading-model',
      statusText: `Loading model ${modelId}...`,
      progress: 0,
    })

    const extractor = await getExtractor(modelId, dtype, (event) => {
      if (typeof event?.progress !== 'number') return
      const normalized = event.progress > 1 ? event.progress / 100 : event.progress
      post(runId, {
        type: 'progress',
        stage: 'loading-model',
        statusText: event.file ? `Loading model file: ${event.file}` : 'Loading model assets...',
        progress: Math.max(0, Math.min(0.2, normalized * 0.2)),
      })
    })

    assertActive(jobId)

    const totalCount = sentences.length

    if (totalCount === 1) {
      post(runId, {
        type: 'progress',
        stage: 'embedding',
        statusText: 'Embedding 1 sentence...',
        progress: 0.4,
      })

      const single = await extractor([sentences[0]], {
        pooling: 'mean',
        normalize: true,
      })

      const dim = single?.dims?.[1] ?? single?.data?.length ?? 0
      const actualTargetDim = Math.max(1, Math.min(dim, Number(targetDim) || dim))

      post(runId, {
        type: 'points-reset',
        points: [{ x: 0, y: 0, sentenceIndex: 0 }],
      })
      post(runId, {
        type: 'progress',
        stage: 'done',
        statusText: 'Done. Embedded 1 sentence.',
        progress: 1,
        embeddingDim: dim,
        usedDim: actualTargetDim,
      })
      post(runId, {
        type: 'done',
        totalCount: 1,
        elapsedMs: performance.now() - started,
      })
      return
    }

    const fitSampleCount = Math.min(umapFitSampleSize, totalCount)
    const sampledIndices = sampleIndices(totalCount, fitSampleCount)
    const sampleMask = new Uint8Array(totalCount)
    for (const index of sampledIndices) sampleMask[index] = 1

    const totalBatches = Math.ceil(totalCount / embeddingBatchSize)
    const totalTransformCount = totalCount - fitSampleCount

    const neighbors = Math.max(2, Math.min(15, totalCount - 1))
    const umap = new UMAP({
      nComponents: 2,
      nNeighbors: neighbors,
      minDist: 0.1,
      spread: 1,
      nEpochs: 300,
    })

    const sampleEmbeddings = []
    const sampleSentenceIndices = []
    const transformQueueEmbeddings = []
    const transformQueueSentenceIndices = []

    let embeddingDim = 0
    let usedDim = 0
    let processedCount = 0
    let transformedCount = 0
    let umapFitDone = false

    const updateProgress = (stage, statusText) => {
      const embedRatio = processedCount / totalCount
      const transformRatio = totalTransformCount > 0 ? transformedCount / totalTransformCount : 1
      const fitBonus = umapFitDone ? 0.1 : 0

      post(runId, {
        type: 'progress',
        stage,
        statusText,
        embeddingDim,
        usedDim,
        progress: Math.min(1, 0.2 + embedRatio * 0.45 + fitBonus + transformRatio * 0.25),
      })
    }

    const runTransformBatch = async (size) => {
      if (!umapFitDone || size <= 0) return

      const indexBatch = transformQueueSentenceIndices.splice(0, size)
      const embeddingBatch = transformQueueEmbeddings.splice(0, size)
      if (!indexBatch.length) return

      updateProgress('umap-transform', `UMAP transform (${transformedCount + indexBatch.length}/${totalTransformCount})`)

      const projectedBatch = await new Promise((resolve) => {
        setTimeout(() => resolve(umap.transform(embeddingBatch)), 0)
      })

      assertActive(jobId)

      post(runId, {
        type: 'points-append',
        points: projectedBatch.map(([x, y], i) => ({
          x,
          y,
          sentenceIndex: indexBatch[i],
        })),
      })

      transformedCount += indexBatch.length
      updateProgress('umap-transform', `UMAP transform (${transformedCount}/${totalTransformCount})`)
      await sleepFrame()
      assertActive(jobId)
    }

    const runFitIfReady = async () => {
      if (umapFitDone || sampleEmbeddings.length < fitSampleCount) return

      updateProgress('umap-fit', `Running UMAP fit on random ${fitSampleCount} points...`)

      const fitProjected = await new Promise((resolve) => {
        setTimeout(() => resolve(umap.fit(sampleEmbeddings)), 0)
      })

      assertActive(jobId)

      post(runId, {
        type: 'points-reset',
        points: fitProjected.map(([x, y], i) => ({
          x,
          y,
          sentenceIndex: sampleSentenceIndices[i],
        })),
      })

      umapFitDone = true
      updateProgress('umap-fit', `UMAP fit completed (${fitSampleCount} points).`)
      await sleepFrame()
      assertActive(jobId)
    }

    for (let batchIndex = 0; batchIndex < totalBatches; batchIndex += 1) {
      assertActive(jobId)

      const start = batchIndex * embeddingBatchSize
      const end = Math.min(start + embeddingBatchSize, totalCount)
      const sentenceBatch = sentences.slice(start, end)

      updateProgress('embedding', `Embedding batch ${batchIndex + 1}/${totalBatches} (${end}/${totalCount})`)

      const output = await extractor(sentenceBatch, {
        pooling: 'mean',
        normalize: true,
      })

      assertActive(jobId)

      const dims = output.dims ?? []
      const batchRows = dims.length >= 2 ? dims[0] : sentenceBatch.length
      const dim = dims.length >= 2 ? dims[1] : output.data.length
      const flat = Array.from(output.data)

      if (!embeddingDim) embeddingDim = dim
      const actualTargetDim = Math.max(1, Math.min(dim, Number(targetDim) || dim))
      usedDim = actualTargetDim

      for (let row = 0; row < batchRows; row += 1) {
        const globalIndex = start + row
        const offset = row * dim
        const vector = flat.slice(offset, offset + actualTargetDim)

        if (sampleMask[globalIndex]) {
          sampleEmbeddings.push(vector)
          sampleSentenceIndices.push(globalIndex)
        } else {
          transformQueueEmbeddings.push(vector)
          transformQueueSentenceIndices.push(globalIndex)
        }
      }

      processedCount = end
      updateProgress('embedding', `Embedding batch ${batchIndex + 1}/${totalBatches} (${end}/${totalCount})`)

      await runFitIfReady()
      while (umapFitDone && transformQueueEmbeddings.length >= umapTransformBatchSize) {
        await runTransformBatch(umapTransformBatchSize)
      }
    }

    await runFitIfReady()
    while (umapFitDone && transformQueueEmbeddings.length > 0) {
      await runTransformBatch(Math.min(umapTransformBatchSize, transformQueueEmbeddings.length))
    }

    assertActive(jobId)

    post(runId, {
      type: 'progress',
      stage: 'done',
      statusText: `Done. Embedded ${totalCount} sentences.`,
      embeddingDim,
      usedDim,
      progress: 1,
    })

    post(runId, {
      type: 'done',
      totalCount,
      elapsedMs: performance.now() - started,
    })
  } catch (error) {
    if (error?.message === '__CANCELLED__') return
    post(runId, {
      type: 'error',
      message: error?.message ?? 'Unknown worker error',
    })
  }
}

self.onmessage = (event) => {
  const message = event.data
  if (message?.type !== 'run') return

  activeJobId += 1
  const jobId = activeJobId
  runJob(message, jobId)
}
