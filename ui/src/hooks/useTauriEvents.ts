import { useEffect } from "react";
import { listen } from "@tauri-apps/api/event";
import type { Job } from "../types";
import { useJobStore } from "../Store/useJobStore";

interface UseTauriEventsProps {
  setJobs: React.Dispatch<React.SetStateAction<Job[]>>;
  setLogs: React.Dispatch<React.SetStateAction<string[]>>;
  setActiveJob: React.Dispatch<React.SetStateAction<Job | null>>;
  setLoadingModel: React.Dispatch<React.SetStateAction<boolean>>;
  addToast: (message: string, type: "success" | "error" | "info" | "warning") => void;
}

export const useTauriEvents = ({
  setJobs,
  setLogs,
  setActiveJob
}: UseTauriEventsProps) => {

  const { setStats, setProgress } = useJobStore();

  useEffect(() => {
    let unlistenProgress: (() => void) | undefined;
    let unlistenStats: (() => void) | undefined;
    let isCleanedUp = false;

    const setup = async () => {
      // 1. Progress Listener
      unlistenProgress = await listen("upscale-progress", (event: any) => {
        if (isCleanedUp) return;
        const { jobId, progress, message, outputPath, eta } = event.payload;

        // Parse Frame Counts from message "Processing Frame X/Y"
        let framesProcessed = 0;
        let totalFrames = 0;
        const frameMatch = message.match(/Frame\s+(\d+)\/(\d+)/);
        if (frameMatch) {
            framesProcessed = parseInt(frameMatch[1]);
            totalFrames = parseInt(frameMatch[2]);
            // Update Global Store
            setProgress(progress, framesProcessed, totalFrames);
        } else if (progress === 100) {
            setProgress(100, 0, 0);
        }

        const updateJobLogic = (j: Job): Job => {
          const isIdMatch = j.id === jobId;
          // Also match generic 'active' jobs if we are running
          const isActiveTarget = (jobId === 'active' || jobId === 'export') &&
                                 (j.status === 'running');

          if (isIdMatch || isActiveTarget) {
             const isFinished = progress === 100 && !!outputPath;
             return {
                 ...j,
                 progress,
                 framesProcessed: framesProcessed || j.framesProcessed,
                 totalFrames: totalFrames || j.totalFrames,
                 statusMessage: message,
                 ...(outputPath ? { outputPath } : {}),
                 eta: eta ?? 0,
                 status: isFinished ? 'done' : j.status
             };
          }
          return j;
        };

        setJobs((prev: Job[]) => prev.map(updateJobLogic));
        setActiveJob((prev) => {
          if (!prev) return null;
          const updated = updateJobLogic(prev);
          if (updated !== prev) return updated;
          return prev;
        });

        // Log significant events
        if (progress === 0 || progress === 100 || (framesProcessed > 0 && framesProcessed % 50 === 0)) {
           setLogs((prev: string[]) => [...prev, `[GPU] ${message} (${progress}%)`]);
        }
      });

      // 2. Stats Listener
      unlistenStats = await listen("system-stats", (event: any) => {
         if (isCleanedUp) return;
         setStats(event.payload);
      });
    };

    setup();

    return () => {
      isCleanedUp = true;
      if (unlistenProgress) unlistenProgress();
      if (unlistenStats) unlistenStats();
    };
  }, [setJobs, setLogs, setActiveJob, setStats, setProgress]);
};
