import React, { useState, useEffect, useMemo } from 'react';

interface LogsPanelProps {
  logs: string[];
  logsEndRef: any;
  setLogs: React.Dispatch<React.SetStateAction<string[]>>;
  darkMode: boolean;
}

// Icons
const IconChevron = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

const IconClear = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
);

const IconActivity = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
  </svg>
);

// Toggle switch component
const ToggleSwitch: React.FC<{ checked: boolean; onChange: () => void; label: string }> = ({ checked, onChange, label }) => (
  <label style={{
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    cursor: 'pointer',
    padding: '2px 8px',
    borderRadius: '4px',
    background: checked ? 'rgba(168,85,247,0.1)' : 'transparent',
    border: checked ? '1px solid rgba(168,85,247,0.3)' : '1px solid transparent',
    transition: 'all 0.15s ease'
  }}>
    <div style={{
      width: '24px',
      height: '14px',
      borderRadius: '7px',
      background: checked ? '#a855f7' : 'rgba(255,255,255,0.1)',
      position: 'relative',
      transition: 'all 0.2s ease'
    }}>
      <div style={{
        width: '10px',
        height: '10px',
        borderRadius: '50%',
        background: checked ? '#fff' : '#666',
        position: 'absolute',
        top: '2px',
        left: checked ? '12px' : '2px',
        transition: 'left 0.2s ease'
      }} />
    </div>
    <span style={{
      fontSize: '9px',
      color: checked ? '#a855f7' : 'var(--text-muted)',
      fontWeight: 600,
      letterSpacing: '0.05em'
    }}>{label}</span>
  </label>
);

export const LogsPanel: React.FC<LogsPanelProps> = ({ logs, logsEndRef, setLogs }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [showVerbose, setShowVerbose] = useState(false);

  // Auto-expand on Error
  useEffect(() => {
    if (logs.length > 0) {
      const lastLog = logs[logs.length - 1];
      if (lastLog.includes("[ERROR]")) {
        setIsCollapsed(false);
      }
    }
  }, [logs]);

  const filteredLogs = useMemo(() => {
    return logs.filter(log => {
      if (!showVerbose && log.includes("[GPU]")) return false;
      return true;
    });
  }, [logs, showVerbose]);

  const lastMessage = filteredLogs.length > 0 ? filteredLogs[filteredLogs.length - 1] : "WAITING...";

  // Count by type
  const errorCount = logs.filter(l => l.includes("[ERROR]")).length;
  const systemCount = logs.filter(l => l.includes("[SYSTEM]")).length;

  // Format log entry
  const formatLogEntry = (log: string, index: number) => {
    const isError = log.includes("[ERROR]");
    const isGPU = log.includes("[GPU]");
    const isSystem = log.includes("[SYSTEM]");
    const isWarning = log.includes("[WARN]");

    // Determine styling
    let accentColor = 'transparent';
    let bgColor = index % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)';
    let textColor = 'var(--text-secondary)';
    let tagColor = 'var(--text-muted)';
    let tagBg = 'transparent';

    if (isError) {
      accentColor = '#ef4444';
      bgColor = 'rgba(239,68,68,0.05)';
      textColor = '#fca5a5';
      tagColor = '#ef4444';
      tagBg = 'rgba(239,68,68,0.15)';
    } else if (isGPU) {
      accentColor = 'var(--brand-primary)';
      textColor = 'var(--brand-primary)';
      tagColor = 'var(--brand-primary)';
      tagBg = 'rgba(0,255,136,0.1)';
    } else if (isSystem) {
      accentColor = '#3b82f6';
      textColor = 'var(--text-primary)';
      tagColor = '#3b82f6';
      tagBg = 'rgba(59,130,246,0.1)';
    } else if (isWarning) {
      accentColor = '#eab308';
      textColor = '#fbbf24';
      tagColor = '#eab308';
      tagBg = 'rgba(234,179,8,0.1)';
    }

    // Extract tag from log
    const tagMatch = log.match(/\[(ERROR|GPU|SYSTEM|WARN|INFO)\]/);
    const tag = tagMatch ? tagMatch[1] : null;
    const cleanLog = tag ? log.replace(`[${tag}]`, '').trim() : log;

    const time = new Date().toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });

    return (
      <div key={index} style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '10px',
        padding: '6px 12px',
        backgroundColor: bgColor,
        borderLeft: `2px solid ${accentColor}`,
        transition: 'background-color 0.15s ease'
      }}>
        {/* Timestamp */}
        <span style={{
          fontSize: '9px',
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
          opacity: 0.6,
          minWidth: '52px',
          flexShrink: 0
        }}>{time}</span>

        {/* Tag badge */}
        {tag && (
          <span style={{
            fontSize: '8px',
            fontWeight: 700,
            padding: '2px 5px',
            borderRadius: '3px',
            background: tagBg,
            color: tagColor,
            letterSpacing: '0.03em',
            minWidth: '42px',
            textAlign: 'center',
            flexShrink: 0
          }}>{tag}</span>
        )}

        {/* Log message */}
        <span style={{
          fontSize: '10px',
          color: textColor,
          fontFamily: 'var(--font-mono)',
          wordBreak: 'break-word',
          lineHeight: 1.5
        }}>{cleanLog}</span>
      </div>
    );
  };

  return (
    <div style={{
      backgroundColor: 'var(--panel-bg)',
      color: 'var(--text-secondary)',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: 'var(--font-mono)',
      fontSize: '10px',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          background: 'rgba(0,0,0,0.2)',
          borderBottom: isCollapsed ? 'none' : '1px solid rgba(255,255,255,0.06)',
          cursor: 'pointer',
          userSelect: 'none',
          gap: '12px'
        }}
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1, minWidth: 0 }}>
          {/* Collapse indicator */}
          <div style={{
            transform: isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s ease',
            color: 'var(--text-muted)'
          }}>
            <IconChevron />
          </div>

          {/* Icon and title */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: '#a855f7'
          }}>
            <IconActivity />
            <span style={{
              fontWeight: 700,
              letterSpacing: '0.05em',
              fontSize: '10px',
              color: 'var(--text-secondary)'
            }}>ACTIVITY</span>
          </div>

          {/* Counts */}
          <div style={{ display: 'flex', gap: '6px' }}>
            {errorCount > 0 && (
              <span style={{
                fontSize: '8px',
                fontWeight: 700,
                padding: '2px 6px',
                borderRadius: '8px',
                background: 'rgba(239,68,68,0.15)',
                color: '#ef4444'
              }}>{errorCount} ERROR{errorCount > 1 ? 'S' : ''}</span>
            )}
            <span style={{
              fontSize: '8px',
              fontWeight: 600,
              padding: '2px 6px',
              borderRadius: '8px',
              background: 'rgba(255,255,255,0.05)',
              color: 'var(--text-muted)'
            }}>{logs.length} ENTRIES</span>
          </div>

          {/* Collapsed preview */}
          {isCollapsed && (
            <span style={{
              opacity: 0.5,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              flex: 1,
              fontSize: '9px'
            }}>
              › {lastMessage.substring(0, 60)}{lastMessage.length > 60 ? '...' : ''}
            </span>
          )}
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }} onClick={(e) => e.stopPropagation()}>
          <ToggleSwitch
            checked={showVerbose}
            onChange={() => setShowVerbose(!showVerbose)}
            label="VERBOSE"
          />

          {logs.length > 0 && (
            <button
              onClick={() => setLogs([])}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                padding: '3px 8px',
                background: 'transparent',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '4px',
                color: 'var(--text-muted)',
                fontSize: '9px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'rgba(239,68,68,0.3)';
                e.currentTarget.style.color = '#ef4444';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
                e.currentTarget.style.color = 'var(--text-muted)';
              }}
            >
              <IconClear />
              CLEAR
            </button>
          )}
        </div>
      </div>

      {/* Log Content */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        display: isCollapsed ? 'none' : 'flex',
        flexDirection: 'column',
      }}>
        {filteredLogs.length === 0 ? (
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px',
            gap: '8px'
          }}>
            <div style={{
              color: 'var(--text-muted)',
              opacity: 0.4,
              fontSize: '11px'
            }}>
              <IconActivity />
            </div>
            <span style={{
              color: 'var(--text-muted)',
              opacity: 0.4,
              fontSize: '10px'
            }}>
              Waiting for events...
            </span>
          </div>
        ) : (
          filteredLogs.map((log, i) => formatLogEntry(log, i))
        )}
        <div ref={logsEndRef} />
      </div>
    </div>
  );
};
