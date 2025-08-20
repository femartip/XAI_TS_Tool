import React, { useState, useEffect } from "react";
import "./styles.css";
import "./HomePage.css";
import { TrainSetting } from "./TrainSetting";
import ImportPage from "./ImportPage";
import axios from 'axios';

export default () => {
    const [datasets, setDatasets] = useState([]);
    const [datasetName, setDatasetName] = useState("");
    const [instanceNumber, setInstanceNumber] = useState(0);
    const [simpMethod, setSimpMethod] = useState("RDP");
    const [alphaValue, setAlphaValue] = useState(0);
    const [currentPage, setCurrentPage] = useState('home');

    useEffect(() => {
        const fetchDatasets = async () => {
            try {
                const response = await axios.get('http://localhost:8000/datasets');
                setDatasets(response.data);
                if (response.data.length > 0) {
                    setDatasetName(response.data[0]);
                }
            } catch (error) {
                console.error("Error fetching datasets:", error);
            }
        };
        fetchDatasets();
    }, []);

    const setDatasetNameFunc = (name) => {
        setDatasetName(name);
    }
    const setInstanceNumberFunc = (number) => {
        setInstanceNumber(number);
    }

    const setSimplificationMethod = (name) => {
        setSimpMethod(name);
    }

    const setAlphaValueFunc = (number) => {
        setAlphaValue(number);
    }

    return (
        <div className="App">
            {currentPage === 'home' ? (
                <div className="home-page">
                    <h1>Interactive XAI Tool</h1>
                    <button className="button-nav" onClick={() => setCurrentPage('import')}>
                        Go to Import Page
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
                            <input type="number" defaultValue={instanceNumber} onChange={(event) => setInstanceNumberFunc(event.target.value)} />
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
                            <h3>Alpha Value</h3>
                            <input
                                type="number"
                                defaultValue={alphaValue}
                                onChange={(event) => setAlphaValueFunc(event.target.value)}
                                step="0.01"
                                min="0"
                                max="1"
                            />
                        </div>
                    </div>

                    <div className="InteractiveTool">
                        {datasetName &&
                            <TrainSetting datasetName={datasetName} instanceNumber={instanceNumber} simpMethod={simpMethod} alphaValue={alphaValue} />}
                    </div>
                </div>
            ) : (
                <div>
                    <button className="button-nav" onClick={() => setCurrentPage('home')} style={{ margin: '10px' }}>
                        Back to Home
                    </button>
                    <ImportPage />
                </div>
            )}
        </div>
    );
};


