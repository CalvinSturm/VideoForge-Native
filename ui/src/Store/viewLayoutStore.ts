import { create } from 'zustand';

// Explicitly export the type
export type PanelId = 'SETTINGS' | 'PREVIEW' | 'QUEUE' | 'ACTIVITY';

interface ViewLayoutState {
  panels: Record<PanelId, boolean>;

  // Actions
  togglePanel: (id: PanelId) => void;
  openPanel: (id: PanelId) => void;
  closePanel: (id: PanelId) => void;
  setAllPanels: (panels: Record<PanelId, boolean>) => void;
  showAllPanels: () => void;
  resetLayout: () => void;
}

const DEFAULT_PANELS: Record<PanelId, boolean> = {
  SETTINGS: true,
  PREVIEW: true,
  QUEUE: true,
  ACTIVITY: true,
};

export const useViewLayoutStore = create<ViewLayoutState>((set) => ({
  panels: { ...DEFAULT_PANELS },

  togglePanel: (id) => set((state) => ({
    panels: {
      ...state.panels,
      [id]: !state.panels[id]
    }
  })),

  openPanel: (id) => set((state) => ({
    panels: {
      ...state.panels,
      [id]: true
    }
  })),

  closePanel: (id) => set((state) => ({
    panels: {
      ...state.panels,
      [id]: false
    }
  })),

  setAllPanels: (newPanels) => set({ panels: newPanels }),

  showAllPanels: () => set({
    panels: { SETTINGS: true, PREVIEW: true, QUEUE: true, ACTIVITY: true }
  }),

  resetLayout: () => set({
    panels: { ...DEFAULT_PANELS }
  }),
}));
