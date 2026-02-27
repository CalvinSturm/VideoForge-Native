/**
 * PanelErrorBoundary — catches render errors in mosaic panels and displays
 * a recovery UI instead of crashing the entire application.
 *
 * Usage:
 *   <PanelErrorBoundary panelId="PREVIEW">
 *     <PreviewPanel ... />
 *   </PanelErrorBoundary>
 *
 * React error boundaries must be class components (React 18).
 */
import React from "react";

interface Props {
    panelId: string;
    children: React.ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

export class PanelErrorBoundary extends React.Component<Props, State> {
    constructor(props: Props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, info: React.ErrorInfo) {
        console.error(
            `[PanelErrorBoundary] ${this.props.panelId} crashed:`,
            error,
            info.componentStack
        );
    }

    private handleRetry = () => {
        this.setState({ hasError: false, error: null });
    };

    render() {
        if (this.state.hasError) {
            return (
                <div
                    style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        height: "100%",
                        padding: "24px",
                        gap: "12px",
                        color: "var(--text-secondary, #999)",
                        fontFamily: "var(--font-mono, monospace)",
                        fontSize: "13px",
                        textAlign: "center",
                    }}
                >
                    <div
                        style={{
                            fontSize: "32px",
                            lineHeight: 1,
                            marginBottom: "4px",
                            opacity: 0.6,
                        }}
                    >
                        ⚠
                    </div>
                    <div style={{ fontWeight: 600, color: "var(--text-primary, #ddd)" }}>
                        {this.props.panelId} panel crashed
                    </div>
                    <div style={{ maxWidth: "320px", opacity: 0.7 }}>
                        {this.state.error?.message || "An unexpected error occurred."}
                    </div>
                    <button
                        onClick={this.handleRetry}
                        style={{
                            marginTop: "8px",
                            padding: "6px 16px",
                            borderRadius: "6px",
                            border: "1px solid var(--border-color, #333)",
                            background: "var(--surface-2, #1a1a2e)",
                            color: "var(--text-primary, #ddd)",
                            cursor: "pointer",
                            fontSize: "12px",
                            fontWeight: 500,
                            transition: "all 0.15s ease",
                        }}
                        onMouseEnter={(e) =>
                        (e.currentTarget.style.background =
                            "var(--accent-color, #3b82f6)")
                        }
                        onMouseLeave={(e) =>
                        (e.currentTarget.style.background =
                            "var(--surface-2, #1a1a2e)")
                        }
                    >
                        Retry
                    </button>
                </div>
            );
        }
        return this.props.children;
    }
}
