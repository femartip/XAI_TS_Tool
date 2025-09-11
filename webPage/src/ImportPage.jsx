import React, { useState, useEffect } from 'react';
import './ImportPage.css';
import { API_HTTP_BASE, WS_BASE } from './config';

const ImportPage = ({ sessionId, onUploadComplete }) => {
    const [modelFile, setModelFile] = useState(null);
    const [datasetFile, setDatasetFile] = useState(null);
    const [datasetName, setDatasetName] = useState('');
    const [scale, setScale] = useState('');
    const [offset, setOffset] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [statusMessage, setStatusMessage] = useState('');
    const [taskId, setTaskId] = useState(null);
    const [startTime, setStartTime] = useState(null);
    const [timeRemaining, setTimeRemaining] = useState(null);

    useEffect(() => {
        if (taskId && isUploading) {
            //const ws = new WebSocket(`ws://localhost:8000/ws/progress/${taskId}`);
            const ws = new WebSocket(`${WS_BASE}/ws/progress/${taskId}`);
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                setProgress(data.progress);
                setStatusMessage(data.message);

                if (startTime && data.progress > 0) {
                    const elapsedTime = (new Date() - startTime) / 1000; // in seconds
                    const totalTime = (elapsedTime / data.progress) * 100;
                    const remaining = totalTime - elapsedTime;
                    setTimeRemaining(remaining);
                }

                if (data.status === 'completed' || data.status === 'error') {
                    setIsUploading(false);
                    setTimeRemaining(null);
                    ws.close();
                    if (data.status === 'completed') {
                        onUploadComplete();
                    }
                }
            };

            return () => {
                ws.close();
            };
        }
    }, [taskId, isUploading, startTime, onUploadComplete]);

    const handleFileSelect = (event, type) => {
        const file = event.target.files[0];
        if (type === 'model') {
            setModelFile(file);
        } else {
            setDatasetFile(file);
        }
    };

    const handleSubmit = async () => {
        if (modelFile && datasetFile && datasetName.trim() && scale !== '' && offset !== '') {
            setIsUploading(true);
            setProgress(0);
            setStatusMessage('Starting upload...');
            setStartTime(new Date());
            setTimeRemaining(null);

            const formData = new FormData();
            formData.append('model_file', modelFile);
            formData.append('dataset_file', datasetFile);
            formData.append('dataset_name', datasetName);
            formData.append('session_id', sessionId);
            formData.append('scale', String(scale));
            formData.append('offset', String(offset));

            try {
                //const response = await fetch('http://localhost:8000/upload', {
                const response = await fetch(`${API_HTTP_BASE}/upload`, {
                    method: 'POST',
                    body: formData,
                });

                const result = await response.json();

                if (response.ok) {
                    setTaskId(result.task_id);
                } else {
                    alert(result.detail || 'Upload failed');
                    setStatusMessage("Error");
                    setIsUploading(false);
                }
            } catch (error) {
                alert('Upload failed');
                setStatusMessage("Error");
                setIsUploading(false);
            }
        }
    };

    const formatTime = (seconds) => {
        if (seconds === null || seconds < 0) return '';
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}m ${remainingSeconds}s remaining`;
    };

    return (
        <div className="import-page">
            <h1>📁 Import Files</h1>

            <div className="upload-grid">
                <div className="upload-card">
                    <h3 className="field-title">
                        <span>🤖 Model File</span>
                        <span className="help" tabIndex={0}>
                            ?
                            <span className="tooltip">Upload a model file (.pth or .pkl) trained on normalized data different from the one to be uploaded.</span>
                        </span>
                    </h3>
                    <input
                        type="file"
                        accept=".pth,.pkl"
                        onChange={(e) => handleFileSelect(e, 'model')}
                        className="file-input"
                    />
                    <p>.pth or .pkl files</p>
                    {modelFile && <div className="file-selected">✅ {modelFile.name}</div>}
                </div>

                <div className="upload-card">
                    <h3 className="field-title">
                        <span>📊 Dataset File</span>
                        <span className="help" tabIndex={0}>
                            ?
                            <span className="tooltip">Upload your dataset as a .npy array. Data is expected to be normalized and different from the one used for training. First column should be labels, remaining columns the time series.</span>
                        </span>
                    </h3>
                    <input
                        type="file"
                        accept=".npy"
                        onChange={(e) => handleFileSelect(e, 'dataset')}
                        className="file-input"
                    />
                    <p>.npy files only</p>
                    {datasetFile && <div className="file-selected">✅ {datasetFile.name}</div>}
                </div>
            </div>

            <div className="dataset-name-section">
                <h3 className="field-title">
                    <span>📝 Dataset Name</span>
                    <span className="help" tabIndex={0}>
                        ?
                        <span className="tooltip">A unique name for this dataset within your session.</span>
                    </span>
                </h3>
                <input
                    type="text"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    placeholder="Enter dataset name..."
                    className="dataset-name-input"
                />
            </div>

            <div className="transform-section">
                <h3 className="transform-title">📐Normalization Parameters</h3>
                <div className="transform-field">
                    <h3 className="field-title">
                        <span>Scale</span>
                        <span className="help" tabIndex={0}>
                            ?
                            <span className="tooltip">MinMax scaling factor used during normalization (typically 1 / (max - min)). For no transformation Scale = 1.0</span>
                        </span>
                    </h3>
                    <input
                        type="number"
                        step="any"
                        value={scale}
                        onChange={(e) => setScale(e.target.value)}
                        placeholder="e.g., 0.12345"
                        className="dataset-name-input"
                    />
                </div>
                <div className="transform-field">
                    <h3 className="field-title">
                        <span>Offset</span>
                        <span className="help" tabIndex={0}>
                            ?
                            <span className="tooltip">Minimum value (offset) used during normalization. Original x = normalized/scale + offset. For no transformation Offset = 0.0</span>
                        </span>
                    </h3>
                    <input
                        type="number"
                        step="any"
                        value={offset}
                        onChange={(e) => setOffset(e.target.value)}
                        placeholder="e.g., -3.0144"
                        className="dataset-name-input"
                    />
                </div>
            </div>
            {isUploading && (
                <div className="progress-section">
                    <div className="progress-bar">
                        <div
                            className="progress-fill"
                            style={{ width: `${progress}%` }}
                        >
                            <span className="progress-text">{`${Math.round(progress)}%`}</span>
                        </div>
                    </div>
                    <p className="progress-message">{statusMessage}</p>
                    <p className="time-remaining">{formatTime(timeRemaining)}</p>
                </div>
            )}
            {!isUploading && statusMessage && (
                <div className={`status-alert ${statusMessage.includes('successfully') ? 'status-success pulse' :
                    statusMessage.includes('Error') || statusMessage.includes('failed') ? 'status-error' : ''}`}>
                    {statusMessage.includes('successfully') ? '🎉 ' : statusMessage.includes('Error') ? '❌ ' : ''}
                    {statusMessage}
                </div>
            )}
            <button
                className="upload-btn"
                onClick={handleSubmit}
                disabled={!modelFile || !datasetFile || !datasetName.trim() || scale === '' || offset === '' || isUploading}
            >
                {isUploading ? 'Processing...' : '🚀 Upload Files'}
            </button>
        </div>
    );
};

export default ImportPage;
