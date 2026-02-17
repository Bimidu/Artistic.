import api from './api';

export const predictionService = {
    async testConnection() {
        const response = await api.get('/health');
        return response.data;
    },

    async predictFromAudio(file, participantId, modelName, useFusion) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('participant_id', participantId || 'CHI');
        if (modelName) formData.append('model_name', modelName);
        formData.append('use_fusion', useFusion);

        const response = await api.post('/predict/audio', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    },

    async predictFromText(text, participantId, modelName, useFusion) {
        const response = await api.post('/predict/text', {
            text,
            participant_id: participantId || 'CHI',
            model_name: modelName,
            use_fusion: useFusion,
        });
        return response.data;
    },

    async predictFromChatFile(file, participantId, modelName, useFusion) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('participant_id', participantId || 'CHI');
        if (modelName) formData.append('model_name', modelName);
        formData.append('use_fusion', useFusion);

        const response = await api.post('/predict/chat_file', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    },

    async getAvailableModels() {
        const response = await api.get('/models');
        return response.data;
    },

    async loadDatasets() {
        const response = await api.get('/training/datasets');
        return response.data;
    },

    async extractFeatures(datasetPaths, component, outputFilename, maxSamples) {
        const response = await api.post('/training/extract-features', {
            dataset_paths: datasetPaths,
            component,
            output_filename: outputFilename,
            max_samples_per_dataset: maxSamples,
        });
        return response.data;
    },

    async startTraining(datasetNames, component, modelTypes, nFeatures) {
        const response = await api.post('/training/train', {
            dataset_names: datasetNames,
            component,
            model_types: modelTypes,
            n_features: nFeatures,
        });
        return response.data;
    },
};
