import React from "react";
import { invoke } from "@tauri-apps/api/core";
import type { Job } from "../types";

interface JobsPanelProps {
  jobs: Job[];
  pauseJob: (id: string) => void;
  cancelJob: (id: string) => void;
  resumeJob: (id: string) => void;
  clearCompleted: () => void;
  showTech: boolean;
}

const formatEta = (seconds?: number) => {
  if (!seconds || seconds <= 0) return null;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  return `${mins}m ${Math.round(seconds % 60)}s`;
};

// --- ICONS ---
const IconCheck = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

const IconX = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const IconClock = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </svg>
);

const IconPlay = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
    <polygon points="5 3 19 12 5 21 5 3" />
  </svg>
);

const IconFolder = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
  </svg>
);

const IconTrash = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
);

const IconStop = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
    <rect x="6" y="6" width="12" height="12" rx="1" />
  </svg>
);

// Spinner with modern styling
const Spinner = () => (
  <div style={{
    width: '16px',
    height: '16px',
    border: '2px solid rgba(0,255,136,0.2)',
    borderTopColor: 'var(--brand-primary)',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite'
  }} />
);

// Status badge component
const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const configs: Record<string, { bg: string; color: string; border: string; label: string }> = {
    done: { bg: 'rgba(16,185,129,0.1)', color: '#10b981', border: 'rgba(16,185,129,0.3)', label: 'COMPLETE' },
    error: { bg: 'rgba(239,68,68,0.1)', color: '#ef4444', border: 'rgba(239,68,68,0.3)', label: 'FAILED' },
    cancelled: { bg: 'rgba(234,179,8,0.1)', color: '#eab308', border: 'rgba(234,179,8,0.3)', label: 'CANCELLED' },
    running: { bg: 'rgba(0,255,136,0.1)', color: 'var(--brand-primary)', border: 'rgba(0,255,136,0.3)', label: 'RUNNING' },
    paused: { bg: 'rgba(168,85,247,0.1)', color: '#a855f7', border: 'rgba(168,85,247,0.3)', label: 'PAUSED' },
    queued: { bg: 'rgba(59,130,246,0.1)', color: '#3b82f6', border: 'rgba(59,130,246,0.3)', label: 'QUEUED' }
  };

  const config = configs[status] || configs.queued;

  return (
    <span style={{
      fontSize: '8px',
      fontWeight: 700,
      padding: '3px 6px',
      borderRadius: '3px',
      background: config.bg,
      color: config.color,
      border: `1px solid ${config.border}`,
      letterSpacing: '0.05em'
    }}>
      {config.label}
    </span>
  );
};

// Action button component
const ActionButton: React.FC<{
  icon: React.ReactNode;
  label?: string;
  onClick: () => void;
  variant?: 'default' | 'danger' | 'success';
  title?: string;
}> = ({ icon, label, onClick, variant = 'default', title }) => {
  const colors = {
    default: { bg: 'rgba(255,255,255,0.03)', border: 'rgba(255,255,255,0.1)', color: 'var(--text-secondary)', hover: 'var(--text-primary)' },
    danger: { bg: 'rgba(239,68,68,0.1)', border: 'rgba(239,68,68,0.3)', color: '#ef4444', hover: '#ff6b6b' },
    success: { bg: 'rgba(0,255,136,0.1)', border: 'rgba(0,255,136,0.3)', color: 'var(--brand-primary)', hover: '#33ff99' }
  };
  const c = colors[variant];

  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '4px',
        padding: label ? '4px 8px' : '4px 6px',
        background: c.bg,
        border: `1px solid ${c.border}`,
        borderRadius: '4px',
        color: c.color,
        fontSize: '9px',
        fontWeight: 600,
        cursor: 'pointer',
        transition: 'all 0.15s ease'
      }}
      onMouseEnter={(e) => e.currentTarget.style.color = c.hover}
      onMouseLeave={(e) => e.currentTarget.style.color = c.color}
    >
      {icon}
      {label}
    </button>
  );
};

export const JobsPanel: React.FC<JobsPanelProps> = ({
  jobs,
  cancelJob,
  clearCompleted,
}) => {
  const isVideo = (path: string) => /\.(mp4|mkv|mov|avi|webm)$/i.test(path);
  const hasCompletedJobs = jobs.some(j => j.status === 'done' || j.status === 'error' || j.status === 'cancelled');

  return (
    <div style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--panel-bg)',
      overflow: 'hidden'
    }}>
      {/* Header with actions */}
      {jobs.length > 0 && (
        <div style={{
          padding: '10px 14px',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'rgba(0,0,0,0.2)'
        }}>
          <span style={{
            fontSize: '10px',
            color: 'var(--text-muted)',
            fontWeight: 600,
            fontFamily: 'var(--font-mono)'
          }}>
            {jobs.filter(j => j.status === 'running').length} ACTIVE / {jobs.length} TOTAL
          </span>

          {hasCompletedJobs && (
            <button
              onClick={clearCompleted}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                background: 'transparent',
                border: '1px solid rgba(255,255,255,0.08)',
                color: 'var(--text-secondary)',
                fontSize: '9px',
                fontWeight: 600,
                padding: '4px 10px',
                borderRadius: '4px',
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.15)';
                e.currentTarget.style.color = 'var(--text-primary)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
                e.currentTarget.style.color = 'var(--text-secondary)';
              }}
            >
              <IconTrash />
              CLEAR COMPLETED
            </button>
          )}
        </div>
      )}

      {/* Empty state */}
      {jobs.length === 0 ? (
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '24px',
          gap: '12px'
        }}>
          <div style={{
            width: '48px',
            height: '48px',
            borderRadius: '12px',
            background: 'rgba(255,255,255,0.03)',
            border: '1px dashed rgba(255,255,255,0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-muted)'
          }}>
            <IconClock />
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              fontSize: '12px',
              fontWeight: 600,
              color: 'var(--text-secondary)',
              marginBottom: '6px'
            }}>
              No Active Jobs
            </div>
            <div style={{
              fontSize: '10px',
              color: 'var(--text-muted)',
              maxWidth: '200px',
              lineHeight: '1.5'
            }}>
              Drop files or configure settings to start processing
            </div>
          </div>
        </div>
      ) : (
        /* Job list */
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '12px',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px'
        }}>
          {jobs.map((job) => {
            const isComplete = job.status === "done" || (job.progress === 100 && job.outputPath);
            const isRunning = job.status === "running" && !isComplete;
            const isError = job.status === "error";
            const isCancelled = job.status === "cancelled";
            const etaText = isRunning ? formatEta(job.eta) : null;
            const filename = job.command.replace(/^(Upscale|Transcode|Export Edited|PREVIEW SAMPLE):\s*/i, "");

            // Determine accent color based on status
            const accentColor = isComplete ? '#10b981' : isError ? '#ef4444' : isCancelled ? '#eab308' : 'var(--brand-primary)';

            return (
              <div
                key={job.id}
                style={{
                  padding: '12px 14px',
                  borderRadius: '8px',
                  background: 'linear-gradient(180deg, #161618, #131315)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderLeft: `3px solid ${accentColor}`,
                  boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '10px',
                  transition: 'all 0.15s ease'
                }}
              >
                {/* Header row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1, minWidth: 0 }}>
                    {/* Status icon */}
                    <div style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: '6px',
                      background: `${accentColor}15`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0
                    }}>
                      {isComplete ? <IconCheck /> : isError ? <IconX /> : isRunning ? <Spinner /> : <IconClock />}
                    </div>

                    {/* Filename */}
                    <span style={{
                      fontWeight: 600,
                      fontSize: '11px',
                      color: 'var(--text-primary)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap'
                    }}>
                      {filename}
                    </span>
                  </div>

                  <StatusBadge status={job.status} />
                </div>

                {/* Progress bar (only for running/queued) */}
                {isRunning && (
                  <div style={{
                    width: '100%',
                    height: '4px',
                    background: 'rgba(0,0,0,0.4)',
                    borderRadius: '2px',
                    overflow: 'hidden',
                    boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.4)'
                  }}>
                    <div style={{
                      width: `${Math.max(job.progress, 2)}%`,
                      height: '100%',
                      background: 'linear-gradient(90deg, var(--brand-primary), #33ff99)',
                      borderRadius: '2px',
                      transition: 'width 0.3s ease-out',
                      boxShadow: '0 0 8px rgba(0,255,136,0.5)'
                    }} />
                  </div>
                )}

                {/* Footer row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {isRunning && (
                      <>
                        <span style={{
                          fontSize: '10px',
                          fontWeight: 600,
                          color: 'var(--brand-primary)',
                          fontFamily: 'var(--font-mono)'
                        }}>
                          {Math.round(job.progress)}%
                        </span>
                        {etaText && (
                          <span style={{
                            fontSize: '9px',
                            color: 'var(--text-muted)',
                            fontFamily: 'var(--font-mono)'
                          }}>
                            ETA: {etaText}
                          </span>
                        )}
                      </>
                    )}
                  </div>

                  {/* Actions */}
                  <div style={{ display: 'flex', gap: '6px' }}>
                    {isRunning && (
                      <ActionButton
                        icon={<IconStop />}
                        label="STOP"
                        onClick={() => cancelJob(job.id)}
                        variant="danger"
                        title="Cancel Job"
                      />
                    )}
                    {isComplete && job.outputPath && (
                      <>
                        <ActionButton
                          icon={<IconPlay />}
                          label={isVideo(job.outputPath) ? "PLAY" : "OPEN"}
                          onClick={() => invoke('open_media', { path: job.outputPath })}
                          variant="success"
                          title="Open Result"
                        />
                        <ActionButton
                          icon={<IconFolder />}
                          onClick={() => invoke('show_in_folder', { path: job.outputPath })}
                          title="Show in Folder"
                        />
                      </>
                    )}
                    {(isComplete || isError || isCancelled) && (
                      <ActionButton
                        icon={<IconX />}
                        onClick={() => cancelJob(job.id)}
                        title="Dismiss"
                      />
                    )}
                  </div>
                </div>

                {/* Error message */}
                {job.errorMessage && (
                  <div style={{
                    fontSize: '9px',
                    color: '#ef4444',
                    fontFamily: 'var(--font-mono)',
                    padding: '6px 8px',
                    background: 'rgba(239,68,68,0.08)',
                    borderRadius: '4px',
                    wordBreak: 'break-all'
                  }}>
                    {job.errorMessage}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <style>{`
        @keyframes spin { 
          from { transform: rotate(0deg); } 
          to { transform: rotate(360deg); } 
        }
      `}</style>
    </div>
  );
};
