import React, { useState, useEffect, useMemo } from 'react';

interface LogsPanelProps {
  logs: string[];
  logsEndRef: any;
  setLogs: React.Dispatch<React.SetStateAction<string[]>>; // Added for type correctness if needed, though unused here
  darkMode: boolean; // Added for prop matching
}

export const LogsPanel: React.FC<LogsPanelProps> = ({ logs, logsEndRef }) => {
  const [isCollapsed, setIsCollapsed] = useState(true);
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
      // Filter out GPU progress unless verbose
      if (!showVerbose && log.includes("[GPU]")) return false;
      return true;
    });
  }, [logs, showVerbose]);

  // Get last relevant message for collapsed view
  const lastMessage = filteredLogs.length > 0 ? filteredLogs[filteredLogs.length - 1] : "IDLE";

  return (
    <div style={{
      backgroundColor: 'var(--panel-bg)',
      color: 'var(--text-secondary)',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: '"JetBrains Mono", monospace',
      fontSize: '10px',
      borderTop: '1px solid var(--panel-border)',
      overflow: 'hidden'
    }}>
      {/* Header / Toggle Bar */}
      <div
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '6px 12px', background: 'rgba(255,255,255,0.02)',
          borderBottom: isCollapsed ? 'none' : '1px solid var(--panel-border)',
          cursor: 'pointer', userSelect: 'none'
        }}
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ transform: isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>▼</span>
          <span style={{ fontWeight: 700, letterSpacing: '0.05em' }}>ACTIVITY LOG</span>
          {isCollapsed && (
            <span style={{ opacity: 0.5, marginLeft: '12px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '300px' }}>
              &gt; {lastMessage}
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: '8px' }} onClick={(e) => e.stopPropagation()}>
           <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
             <input
               type="checkbox"
               checked={showVerbose}
               onChange={(e) => setShowVerbose(e.target.checked)}
               style={{ width: '12px', height: '12px' }}
             />
             <span style={{ fontSize: '9px', opacity: 0.7 }}>VERBOSE</span>
           </label>
        </div>
      </div>

      {/* Log Content */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        display: isCollapsed ? 'none' : 'flex',
        flexDirection: 'column',
      }}>
        {filteredLogs.length === 0 && (
          <div style={{ padding: '12px', opacity: 0.3 }}>
            &gt; WAITING FOR EVENTS...
          </div>
        )}

        {filteredLogs.map((log: string, i: number) => {
          const isError = log.includes("[ERROR]");
          const isGPU = log.includes("[GPU]");
          const isSystem = log.includes("[SYSTEM]");

          let bg = i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)';
          let color = 'var(--text-secondary)';
          let borderLeft = '2px solid transparent';

          if (isError) {
             color = '#ff6b6b';
             bg = 'rgba(239, 68, 68, 0.08)';
             borderLeft = '2px solid #ef4444';
          } else if (isGPU) {
             color = 'var(--brand-primary)';
             borderLeft = '2px solid var(--brand-primary)';
          } else if (isSystem) {
             color = 'var(--text-primary)';
          }

          const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute:'2-digit', second:'2-digit' });

          return (
            <div key={i} style={{
              display: 'flex',
              padding: '4px 8px',
              backgroundColor: bg,
              borderLeft: borderLeft,
              lineHeight: '1.4',
              color: color
            }}>
              <span style={{ opacity: 0.4, marginRight: '8px', minWidth: '50px', display: 'inline-block' }}>{time}</span>
              <span style={{ wordBreak: 'break-all' }}>{log}</span>
            </div>
          );
        })}
        <div ref={logsEndRef} />
      </div>
    </div>
  );
};
