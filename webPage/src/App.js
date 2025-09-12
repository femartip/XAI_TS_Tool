import React, { useState, useEffect } from "react";
import "./styles.css";
import "./HomePage.css";
import { TrainSetting } from "./TrainSetting";
import ImportPage from "./ImportPage";
import axios from 'axios';
import { API_HTTP_BASE } from './config';
import AboutPage from "./AboutPage";

export default () => {
    const [datasets, setDatasets] = useState([]);
    const [datasetName, setDatasetName] = useState("");
    const [instanceNumber, setInstanceNumber] = useState(0);
    const [simpMethod, setSimpMethod] = useState("RDP");
    const [alphaValue, setAlphaValue] = useState(1);
    const [loyaltyValue, setLoyaltyValue] = useState(0);
    const [complexityValue, setComplexityValue] = useState(0);
    const [paramMode, setParamMode] = useState('alpha'); // 'alpha' | 'loyalty' | 'complexity'
    const [currentPage, setCurrentPage] = useState('home');
    const [sessionId, setSessionId] = useState(null);

    const fetchDatasets = async () => {
        if (!sessionId) return;
        try {
            //const response = await axios.get(`http://localhost:8000/datasets?session_id=${sessionId}`);
            const response = await axios.get(`${API_HTTP_BASE}/datasets`, { params: { session_id: sessionId } });
            setDatasets(response.data);
            if (response.data.length > 0) {
                setDatasetName(response.data[0]);
            }
        } catch (error) {
            console.error("Error fetching datasets:", error);
        }
    };

    useEffect(() => {
        const fetchSessionId = async () => {
            try {
                let storedSessionId = localStorage.getItem('sessionId');
                if (storedSessionId) {
                    setSessionId(storedSessionId);
                } else {
                    const response = await axios.get(`${API_HTTP_BASE}/session`);
                    const newSessionId = response.data.session_id;
                    localStorage.setItem('sessionId', newSessionId);
                    setSessionId(newSessionId);
                }
            } catch (error) {
                console.error("Error fetching session ID:", error);
            }
        };
        fetchSessionId();
    }, []);

    useEffect(() => {
        fetchDatasets();
    }, [sessionId]);

    const setDatasetNameFunc = (name) => {
        setDatasetName(name);
    }
    const setInstanceNumberFunc = (number) => {
        setInstanceNumber(number);
    }

    const setSimplificationMethod = (name) => {
        setSimpMethod(name);
    }

    const setAlphaValueFunc = (number) => { setAlphaValue(number); };
    const setLoyaltyValueFunc = (number) => { setLoyaltyValue(number); };
    const setComplexityValueFunc = (number) => { setComplexityValue(number); };

    // Keep non-selected boxes in sync automatically using backend metrics
    useEffect(() => {
        if (!datasetName || !sessionId) return;
        const selectedVal = paramMode === 'alpha' ? alphaValue : (paramMode === 'loyalty' ? loyaltyValue : complexityValue);
        if (selectedVal === "" || isNaN(parseFloat(selectedVal))) return;

        const doFetch = () => {
            const params = {
                simp_algo: simpMethod,
                dataset_name: datasetName,
                session_id: sessionId,
                is_global: ["Chinatown", "ECG200", "ItalyPowerDemand"].includes(datasetName),
                selection_type: paramMode,
                value: selectedVal
            };
            axios.get(`${API_HTTP_BASE}/param_metrics`, { params })
                .then((res) => {
                    const { alpha, loyalty, complexity } = res.data || {};
                    // Only update NON-selected fields to avoid fighting user typing
                    if (paramMode !== 'alpha' && typeof alpha === 'number') setAlphaValue(alpha);
                    if (paramMode !== 'loyalty' && typeof loyalty === 'number') setLoyaltyValue(loyalty);
                    if (paramMode !== 'complexity' && typeof complexity === 'number') setComplexityValue(complexity);
                })
                .catch((err) => {
                    console.error('Failed to fetch param metrics:', err);
                });
        };

        // Debounce a bit to avoid overwriting while typing
        const t = setTimeout(doFetch, 250);
        return () => clearTimeout(t);
        // Trigger whenever user changes selection or value, or when dataset/method change
    }, [paramMode, alphaValue, loyaltyValue, complexityValue, datasetName, simpMethod, sessionId]);

    return (
        <div className="App">
            {currentPage === 'home' ? (
                <div className="home-page">
                    <h1>Interactive XAI Tool</h1>
                    <button className="button-nav" onClick={() => setCurrentPage('import')}>
                        Go to Import Page
                    </button>
                    <button className="button-accent" onClick={() => setCurrentPage('about')} style={{ marginLeft: '10px' }}>
                        Learn about the Tool
                    </button>

                    <div className="control-grid">
                        <div className="control-card">
                            <h3>Dataset</h3>
                            <select value={datasetName} onChange={(event) => setDatasetNameFunc(event.target.value)}>
                                {datasets.map(ds => (
                                    <option key={ds} value={ds}>{ds}</option>
                                ))}
                            </select>
                        </div>

                        <div className="control-card">
                            <h3>Instance Number</h3>
                            <input type="number" defaultValue={instanceNumber} onChange={(event) => setInstanceNumberFunc(event.target.value)} min="0" />
                        </div>

                        <div className="control-card control-card--method">
                            <h3>Simplification Method</h3>
                            <select defaultValue={"RDP"} onChange={(event) => setSimplificationMethod(event.target.value)}>
                                <option value="RDP">RDP</option>
                                <option value="VW">VW</option>
                                <option value="OS">OS</option>
                                <option value="Bottom-up">BU</option>
                                <option value={"LSF"}>LSF</option>
                            </select>
                        </div>

                        <div className="control-card">
                            <h3>Simplification Parameters</h3>
                            <div className="param-grid">
                                <div className="param-box">
                                    <div className="param-title">
                                        <span>Alpha</span>
                                        <span className="help" tabIndex={0}>i
                                            <span className="tooltip">Algorithm parameter [0-1]. Lower alpha == less segments.</span>
                                        </span>
                                    </div>
                                    <label className="param-select">
                                        <input
                                            type="radio"
                                            name="paramMode"
                                            value="alpha"
                                            checked={paramMode === 'alpha'}
                                            onChange={() => setParamMode('alpha')}
                                        /> Select
                                    </label>
                                    <input
                                        type="number"
                                        value={alphaValue}
                                        onChange={(e) => paramMode === 'alpha' ? setAlphaValueFunc(e.target.value) : null}
                                        step="0.01"
                                        min="0"
                                        max="1"
                                        disabled={paramMode !== 'alpha'}
                                    />
                                </div>
                                <div className="param-box">
                                    <div className="param-title">
                                        <span>Loyalty (κ)</span>
                                        <span className="help" tabIndex={0}>i
                                            <span className="tooltip">Target classification loyalty metric measured by Cohen's kappa [0-1].</span>
                                        </span>
                                    </div>
                                    <label className="param-select">
                                        <input
                                            type="radio"
                                            name="paramMode"
                                            value="loyalty"
                                            checked={paramMode === 'loyalty'}
                                            onChange={() => setParamMode('loyalty')}
                                        /> Select
                                    </label>
                                    <input
                                        type="number"
                                        value={loyaltyValue}
                                        onChange={(e) => paramMode === 'loyalty' ? setLoyaltyValueFunc(e.target.value) : null}
                                        step="0.01"
                                        min="0"
                                        max="1"
                                        disabled={paramMode !== 'loyalty'}
                                    />
                                </div>
                                <div className="param-box">
                                    <div className="param-title">
                                        <span>Complexity</span>
                                        <span className="help" tabIndex={0}>i
                                            <span className="tooltip">Target model complexity (number of segments of simplification / total segments) [0-1].</span>
                                        </span>
                                    </div>
                                    <label className="param-select">
                                        <input
                                            type="radio"
                                            name="paramMode"
                                            value="complexity"
                                            checked={paramMode === 'complexity'}
                                            onChange={() => setParamMode('complexity')}
                                        /> Select
                                    </label>
                                    <input
                                        type="number"
                                        value={complexityValue}
                                        onChange={(e) => paramMode === 'complexity' ? setComplexityValueFunc(e.target.value) : null}
                                        step="0.01"
                                        min="0"
                                        max="1"
                                        disabled={paramMode !== 'complexity'}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="InteractiveTool">
                        {datasetName &&
                            <TrainSetting sessionId={sessionId} datasetName={datasetName} instanceNumber={instanceNumber} simpMethod={simpMethod} alphaValue={paramMode === 'alpha' ? alphaValue : (paramMode === 'loyalty' ? loyaltyValue : complexityValue)} selectionType={paramMode} />}
                    </div>
                </div>
            ) : currentPage === 'import' ? (
                <div>
                    <button className="button-nav" onClick={() => setCurrentPage('home')} style={{ margin: '10px' }}>
                        Back to Home
                    </button>
                    <ImportPage sessionId={sessionId} onUploadComplete={fetchDatasets} />
                </div>
            ) : (
                <AboutPage onBack={() => setCurrentPage('home')} />
            )}
        </div>
    );
};
