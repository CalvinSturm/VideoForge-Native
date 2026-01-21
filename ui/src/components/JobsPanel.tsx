import React from "react";
import { invoke } from "@tauri-apps/api/core";
import type { Job } from "../types";

interface JobsPanelProps {
  jobs: Job[];
  pauseJob: (id: string) => void;
  cancelJob: (id: string) => void;
  resumeJob: (id: string) => void;
  clearCompleted: () => void; // New Prop
  showTech: boolean;
}

const formatEta = (seconds?: number) => {
  if (!seconds || seconds <= 0) return null;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  return `${mins}m ${Math.round(seconds % 60)}s`;
};

// --- ICONS ---
const IconCheck = () => <span style={{ color: "var(--brand-primary)" }}>✅</span>;
const IconError = () => <span>❌</span>;
const IconClock = () => <span style={{ color: "var(--text-muted)" }}>⏳</span>;
const IconSpin = () => (
  <div style={{
    width: 12, height: 12, border: "2px solid var(--brand-primary)",
    borderTopColor: "transparent", borderRadius: "50%",
    animation: "spin 1s linear infinite"
  }} />
);

export const JobsPanel: React.FC<JobsPanelProps> = ({
  jobs,
  cancelJob,
  clearCompleted,
  showTech
}) => {

  const getStatusIcon = (status: string, progress: number) => {
    switch (status) {
      case "done": return <IconCheck />;
      case "error": return <IconError />;
      case "cancelled": return <span style={{ color: "#eab308" }}>⛔</span>;
      case "running": return <IconSpin />;
      default: return <IconClock />;
    }
  };

  const isVideo = (path: string) => /\.(mp4|mkv|mov|avi|webm)$/i.test(path);

  const hasCompletedJobs = jobs.some(j => j.status === 'done' || j.status === 'error' || j.status === 'cancelled');

  return (
    <div className="panel-content" style={{ padding: '0', height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--panel-bg)' }}>
      {/* HEADER WITH CLEAR BUTTON */}
      {hasCompletedJobs && (
        <div style={{
          padding: '8px 12px',
          borderBottom: '1px solid var(--panel-border)',
          display: 'flex', justifyContent: 'flex-end',
          background: 'rgba(255,255,255,0.02)'
        }}>
          <button
            onClick={clearCompleted}
            style={{
              background: 'transparent', border: 'none',
              color: 'var(--text-secondary)', fontSize: '10px',
              cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px',
              padding: '4px 8px', borderRadius: '4px'
            }}
            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)'; e.currentTarget.style.color = '#ededed'; }}
            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; e.currentTarget.style.color = 'var(--text-secondary)'; }}
          >
            CLEAR COMPLETED
          </button>
        </div>
      )}

      {jobs.length === 0 ? (
        <div style={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-secondary)',
          opacity: 0.5,
          padding: '12px'
        }}>
          <div style={{ fontSize: '24px', marginBottom: '8px' }}>📭</div>
          <div style={{ fontSize: '11px', fontWeight: 600 }}>NO ACTIVE JOBS</div>
          <div style={{ fontSize: '9px', marginTop: '4px', maxWidth: '180px', textAlign: 'center', lineHeight: '1.4' }}>
            Drag and drop videos or configure settings to begin processing.
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', overflowY: 'auto', padding: '12px' }}>
          {jobs.map((job) => {
            const isComplete = job.status === "done" || (job.progress === 100 && job.outputPath);
            const isRunning = job.status === "running" && !isComplete;
            const isError = job.status === "error";
            const etaText = isRunning ? formatEta(job.eta) : null;
            const filename = job.command.replace(/^(Upscale|Transcode|Export Edited):\s*/, "");
            const fullPath = job.outputPath || job.id;

            return (
              <div key={job.id}
                title={fullPath}
                style={{
                  padding: '12px',
                  borderRadius: '6px',
                  backgroundColor: '#161618',
                  border: '1px solid rgba(255, 255, 255, 0.06)',
                  borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                  position: 'relative',
                  transition: 'transform 0.1s, box-shadow 0.1s',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px'
                }}
                onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-1px)'; e.currentTarget.style.boxShadow = '0 4px 8px rgba(0,0,0,0.4)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.3)'; }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 700, fontSize: '11px', color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '85%', fontFamily: 'var(--font-sans)' }}>
                    {filename}
                  </span>
                  <div style={{ fontSize: '12px' }}>{getStatusIcon(job.status, job.progress)}</div>
                </div>

                {(!isComplete && !isError && job.status !== 'cancelled') && (
                  <div style={{ width: '100%', height: '4px', background: 'rgba(0,0,0,0.5)', borderRadius: '2px', overflow: 'hidden', boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.5)' }}>
                    <div style={{ width: `${Math.max(job.progress, 5)}%`, height: '100%', background: 'var(--brand-primary)', transition: 'width 0.3s ease-out', boxShadow: '0 0 8px rgba(0,255,136,0.5)' }} />
                  </div>
                )}

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', height: '20px' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <span style={{ fontSize: '9px', color: isError ? '#ef4444' : 'var(--text-secondary)', fontWeight: 600, fontFamily: 'var(--font-sans)' }}>
                      {isError ? "FAILED" : (isComplete ? "COMPLETED" : (job.status === 'cancelled' ? "CANCELLED" : `${Math.round(job.progress)}%`))}
                    </span>
                    {etaText && <span style={{ fontSize: '9px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>ETA: {etaText}</span>}
                  </div>

                  <div style={{ display: 'flex', gap: '6px' }}>
                    {isRunning && (
                      <button onClick={() => cancelJob(job.id)} style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#ef4444', fontSize: '9px', padding: '2px 6px', borderRadius: '3px', cursor: 'pointer', fontWeight: 600 }}>STOP</button>
                    )}
                    {(isComplete && job.outputPath) && (
                      <>
                        <button
                          onClick={() => invoke('open_media', { path: job.outputPath })}
                          style={{ background: 'transparent', border: '1px solid var(--panel-border)', color: 'var(--brand-primary)', fontSize: '9px', padding: '2px 6px', borderRadius: '3px', cursor: 'pointer' }}
                          title="Open Media"
                        >
                          {isVideo(job.outputPath!) ? "PLAY" : "OPEN"}
                        </button>
                        <button onClick={() => invoke('show_in_folder', { path: job.outputPath })} style={{ background: 'transparent', border: '1px solid var(--panel-border)', color: 'var(--text-primary)', fontSize: '9px', padding: '2px 6px', borderRadius: '3px', cursor: 'pointer' }} title="Show in Folder">📂</button>
                      </>
                    )}
                    {(isComplete || isError || job.status === "cancelled") && (
                      <button onClick={() => cancelJob(job.id)} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', fontSize: '11px', padding: '2px 4px', cursor: 'pointer' }} title="Dismiss">✕</button>
                    )}
                  </div>
                </div>

                {job.errorMessage && <div style={{ fontSize: '9px', color: '#ef4444', fontFamily: 'monospace', marginTop: '2px', wordBreak: 'break-all' }}>Error: {job.errorMessage}</div>}
              </div>
            );
          })}
        </div>
      )}
      <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
    </div>
  );
};
