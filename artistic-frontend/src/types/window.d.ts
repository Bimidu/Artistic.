export {};

declare global {
  interface Window {
    __artisticInitHomeUi?: () => void;
    __artisticToggleResizeBound?: boolean;

    testConnection?: () => void | Promise<void>;
    predictFromAudio?: () => void | Promise<void>;
    predictFromChatFile?: () => void | Promise<void>;

    loadDatasets?: () => void | Promise<void>;
    extractFeatures?: () => void | Promise<void>;
    loadAvailableDatasetsForTraining?: () => void | Promise<void>;
    startTraining?: () => void | Promise<void>;
    loadAvailableModels?: () => void | Promise<void>;

    askCounterfactualGPT?: () => void | Promise<void>;

    toggleHyperparameters?: () => void;
    simulateCounterfactualChat?: () => void;
    closeModelDetails?: (event?: unknown) => void;
  }
}

