import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'
import { wsUrl } from '../services/api'
import type { FrameResult, StreamMessage } from '../types/api'

export type StreamState = 'idle' | 'starting' | 'running' | 'error'

interface Options {
  confidence: number
  iou: number
  imageSize: number
  tracking: boolean
  streamWidth: number
  jpegQuality: number
}

/**
 * Captures the laptop webcam in the browser and streams JPEG frames to /ws/stream.
 * Exactly one frame is in flight at a time, so a slow CPU backend never builds a backlog;
 * the video element keeps rendering at full camera rate independently of inference.
 */
export function useWebcamStream(videoRef: RefObject<HTMLVideoElement | null>, opts: Options) {
  const [state, setState] = useState<StreamState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<FrameResult | null>(null)
  const [loopFps, setLoopFps] = useState<number | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const lastSentRef = useRef<number>(0)
  const optsRef = useRef(opts)
  optsRef.current = opts

  const sendConfig = useCallback(() => {
    const ws = wsRef.current
    if (ws?.readyState !== WebSocket.OPEN) return
    const o = optsRef.current
    ws.send(
      JSON.stringify({
        type: 'config',
        confidence: o.confidence,
        iou: o.iou,
        image_size: o.imageSize,
        track: o.tracking,
        source: 'webcam',
        label: 'Laptop webcam',
      }),
    )
  }, [])

  // push threshold changes to the running session
  useEffect(sendConfig, [opts.confidence, opts.iou, opts.imageSize, opts.tracking, sendConfig])

  const sendFrame = useCallback(() => {
    const ws = wsRef.current
    const video = videoRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN || !video || !video.videoWidth) return
    const o = optsRef.current
    const scale = Math.min(1, o.streamWidth / video.videoWidth)
    const canvas = (canvasRef.current ??= document.createElement('canvas'))
    canvas.width = Math.round(video.videoWidth * scale)
    canvas.height = Math.round(video.videoHeight * scale)
    canvas.getContext('2d')!.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob(
      (blob) => {
        if (blob && ws.readyState === WebSocket.OPEN) {
          lastSentRef.current = performance.now()
          ws.send(blob)
        }
      },
      'image/jpeg',
      o.jpegQuality,
    )
  }, [videoRef])

  const stop = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setState('idle')
    setLoopFps(null)
  }, [videoRef])

  const start = useCallback(async () => {
    setError(null)
    setResult(null)
    setState('starting')
    try {
      const media = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      streamRef.current = media
      const video = videoRef.current!
      video.srcObject = media
      await video.play()
    } catch (e) {
      setState('error')
      setError(`Camera unavailable: ${e instanceof Error ? e.message : String(e)}`)
      return
    }

    const ws = new WebSocket(wsUrl('/ws/stream'))
    wsRef.current = ws
    ws.onopen = sendConfig
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as StreamMessage
      if (msg.type === 'ready') {
        setState('running')
        sendFrame()
      } else if (msg.type === 'result') {
        const elapsed = performance.now() - lastSentRef.current
        setLoopFps((prev) => {
          const inst = 1000 / elapsed
          return prev == null ? inst : prev * 0.8 + inst * 0.2 // smoothed end-to-end rate
        })
        setResult(msg)
        requestAnimationFrame(sendFrame)
      } else if (msg.type === 'error') {
        setError(msg.message)
        requestAnimationFrame(sendFrame)
      }
    }
    ws.onerror = () => setError('WebSocket error — is the backend running?')
    ws.onclose = () => {
      if (wsRef.current === ws) {
        stop()
        setError((e) => e ?? 'Stream closed by server')
      }
    }
  }, [videoRef, sendConfig, sendFrame, stop])

  const resetTracking = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: 'reset' }))
  }, [])

  useEffect(() => stop, [stop])

  return { state, error, result, loopFps, start, stop, resetTracking }
}
