import { useRef, useState, useCallback, useEffect } from 'react';
import { AudioEngine, DEFAULT_PARAMS } from '../audio/engine';
import type { EngineParams, RetuneStatus } from '../audio/engine';

export function useAudioEngine() {
  const engineRef = useRef<AudioEngine | null>(null);
  const [engine, setEngine] = useState<AudioEngine | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [params, setParams] = useState<EngineParams>(DEFAULT_PARAMS);
  const [retuneStatus, setRetuneStatus] = useState<RetuneStatus | null>(null);
  const animRef = useRef<number>(0);

  const init = useCallback(async () => {
    if (engineRef.current) return;
    const engine = new AudioEngine();
    await engine.init();
    engine.setRetuneStatusCallback(setRetuneStatus);
    engineRef.current = engine;
    setEngine(engine);
    setIsReady(true);
  }, []);

  const loadFile = useCallback(async (arrayBuffer: ArrayBuffer, name: string) => {
    if (!engineRef.current) await init();
    const info = await engineRef.current!.loadAudioBuffer(arrayBuffer);
    setFileName(name);
    setDuration(info.duration);
    setCurrentTime(0);
  }, [init]);

  const play = useCallback(() => {
    engineRef.current?.play(currentTime);
    setIsPlaying(true);
  }, [currentTime]);

  const stop = useCallback(() => {
    if (engineRef.current) {
      setCurrentTime(engineRef.current.currentTime);
      engineRef.current.stop();
    }
    setIsPlaying(false);
  }, []);

  const togglePlay = useCallback(() => {
    if (isPlaying) stop();
    else play();
  }, [isPlaying, play, stop]);

  const restart = useCallback(() => {
    engineRef.current?.restart();
    setCurrentTime(0);
    setIsPlaying(true);
  }, []);

  const seek = useCallback((time: number) => {
    setCurrentTime(time);
    if (isPlaying) {
      engineRef.current?.play(time);
    }
  }, [isPlaying]);

  const updateParams = useCallback((newParams: Partial<EngineParams>) => {
    setParams(prev => {
      const updated = { ...prev, ...newParams };
      engineRef.current?.updateParams(newParams);
      return updated;
    });
  }, []);

  // Animation frame for current time updates
  useEffect(() => {
    const tick = () => {
      if (engineRef.current?.playing) {
        setCurrentTime(engineRef.current.currentTime);
        setIsPlaying(true);
      } else if (isPlaying) {
        setIsPlaying(false);
      }
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [isPlaying]);

  return {
    engine,
    isReady,
    isPlaying,
    fileName,
    duration,
    currentTime,
    params,
    retuneStatus,
    init,
    loadFile,
    play,
    stop,
    togglePlay,
    restart,
    seek,
    updateParams,
  };
}
