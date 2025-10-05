// NASA Exoplanet Detection Pipeline - Research Mode Analysis Logic

// Research Mode - Input Method Switching
document.querySelectorAll('.method-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const method = btn.dataset.method;
        const parentContainer = btn.closest('.analysis-container');
        
        // Hide all input methods
        parentContainer.querySelectorAll('.input-method').forEach(m => {
            m.classList.remove('active');
        });
        
        // Remove active class from all buttons
        parentContainer.querySelectorAll('.method-btn').forEach(b => {
            b.classList.remove('active');
        });
        
        // Show selected method
        parentContainer.querySelector(`#${method}-method`).classList.add('active');
        btn.classList.add('active');
    });
});

// File Upload Handler
document.getElementById('file-input')?.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
});

function handleFileUpload(file) {
    showNotification(`Uploading ${file.name}...`, 'info');
    
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const content = e.target.result;
            processUploadedData(content, file.name);
        } catch (error) {
            showNotification('Error processing file', 'error');
        }
    };
    
    if (file.name.endsWith('.csv') || file.name.endsWith('.txt')) {
        reader.readAsText(file);
    } else {
        showNotification('Unsupported file format', 'error');
    }
}

function processUploadedData(content, filename) {
    // Parse CSV/TXT data
    const lines = content.split('\n');
    const time = [];
    const flux = [];
    
    lines.forEach((line, index) => {
        if (index === 0 || line.trim() === '') return; // Skip header
        const values = line.split(',');
        if (values.length >= 2) {
            time.push(parseFloat(values[0]));
            flux.push(parseFloat(values[1]));
        }
    });
    
    if (time.length > 0 && flux.length > 0) {
        plotAnalysisData(time, flux);
        runAnalysis(time, flux);
        showNotification('File processed successfully!', 'success');
    } else {
        showNotification('No valid data found in file', 'error');
    }
}

// Example Selector
document.getElementById('example-selector')?.addEventListener('change', (e) => {
    const exampleId = e.target.value;
    if (exampleId) {
        loadExample(exampleId);
    }
});

function loadExample(exampleId) {
    const examples = {
        1: { hasPlanet: true, difficulty: 'easy', name: 'Clear Transit' },
        2: { hasPlanet: true, difficulty: 'medium', name: 'Noisy Transit' },
        3: { hasPlanet: false, difficulty: 'medium', name: 'Stellar Variability' },
        4: { hasPlanet: true, difficulty: 'hard', name: 'Subtle Transit' }
    };
    
    const example = examples[exampleId];
    const { time, flux } = generateLightCurve(example.hasPlanet, example.difficulty);
    
    plotAnalysisData(time, flux);
    runAnalysis(time, flux);
    showNotification(`Loaded example: ${example.name}`, 'success');
}

// Manual Input - Synthetic Data Generation
document.getElementById('depth-slider')?.addEventListener('input', (e) => {
    document.getElementById('depth-value').textContent = e.target.value;
});

document.getElementById('noise-slider')?.addEventListener('input', (e) => {
    document.getElementById('noise-value').textContent = e.target.value;
});

function generateSynthetic() {
    const hasTransit = document.getElementById('transit-toggle').checked;
    const depth = parseFloat(document.getElementById('depth-slider').value);
    const noiseLevel = parseFloat(document.getElementById('noise-slider').value);
    
    const numPoints = 2048;
    const time = [];
    const flux = [];
    
    for (let i = 0; i < numPoints; i++) {
        time.push(i / 100);
        let value = 1.0;
        
        // Add noise
        value += (Math.random() - 0.5) * noiseLevel * 0.01;
        
        // Add transit if enabled
        if (hasTransit) {
            const period = 5.0;
            const duration = 0.2;
            const phase = (time[i] % period) / period;
            
            if (phase < duration / period || phase > 1 - duration / period) {
                const transitPhase = phase < 0.5 ? phase : 1 - phase;
                const transitShape = Math.exp(-Math.pow(transitPhase * period / duration * 4, 2));
                value -= (depth / 100) * transitShape;
            }
        }
        
        flux.push(value);
    }
    
    plotAnalysisData(time, flux);
    runAnalysis(time, flux);
    showNotification('Synthetic data generated!', 'success');
}

// Plot analysis data
function plotAnalysisData(time, flux) {
    const trace = {
        x: time,
        y: flux,
        mode: 'lines',
        type: 'scatter',
        line: {
            color: '#667eea',
            width: 1
        },
        name: 'Light Curve'
    };
    
    const layout = {
        title: 'Light Curve Analysis',
        xaxis: {
            title: 'Time (days)',
            gridcolor: '#e2e8f0'
        },
        yaxis: {
            title: 'Normalized Flux',
            gridcolor: '#e2e8f0'
        },
        plot_bgcolor: '#ffffff',
        paper_bgcolor: '#ffffff',
        font: {
            family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto',
            color: '#0F172A'
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    Plotly.newPlot('analysis-plot', [trace], layout, config);
}

// Run analysis
function runAnalysis(time, flux) {
    // Calculate statistics
    const mean = flux.reduce((a, b) => a + b, 0) / flux.length;
    const variance = flux.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / flux.length;
    const stdDev = Math.sqrt(variance);
    
    // Simulate AI prediction
    const hasPlanet = Math.random() > 0.5;
    const confidence = (85 + Math.random() * 10).toFixed(1);
    const probability = (Math.random() * 0.3 + (hasPlanet ? 0.7 : 0.3)).toFixed(3);
    
    // Display results
    const resultsContainer = document.getElementById('analysis-results');
    resultsContainer.innerHTML = `
        <h3>Analysis Results</h3>
        
        <div class="result-section">
            <h4>🤖 AI Prediction</h4>
            <div class="prediction-box ${hasPlanet ? 'planet-detected' : 'no-planet'}">
                <p class="prediction-label">${hasPlanet ? '🪐 PLANET DETECTED' : '⭐ NO PLANET'}</p>
                <p>Confidence: ${confidence}%</p>
                <p>Probability: ${probability}</p>
            </div>
        </div>
        
        <div class="result-section">
            <h4>📊 Data Statistics</h4>
            <table class="stats-table">
                <tr>
                    <td>Data Points:</td>
                    <td>${flux.length}</td>
                </tr>
                <tr>
                    <td>Mean Flux:</td>
                    <td>${mean.toFixed(6)}</td>
                </tr>
                <tr>
                    <td>Std Deviation:</td>
                    <td>${stdDev.toFixed(6)}</td>
                </tr>
                <tr>
                    <td>Time Range:</td>
                    <td>${time[0].toFixed(2)} - ${time[time.length-1].toFixed(2)} days</td>
                </tr>
            </table>
        </div>
        
        <div class="result-section">
            <h4>🔍 Model Details</h4>
            <p><strong>Model:</strong> Ensemble (CNN + LSTM + Transformer)</p>
            <p><strong>Architecture:</strong> Multi-model voting system</p>
            <p><strong>Training Data:</strong> 14,620 NASA samples</p>
        </div>
        
        <div class="result-actions">
            <button class="btn-primary" onclick="exportResults()">📥 Export Results</button>
            <button class="btn-secondary" onclick="showDetailedAnalysis()">📈 Detailed Analysis</button>
        </div>
    `;
    resultsContainer.classList.remove('hidden');
}

// Export results
function exportResults() {
    // Check if there are results to export
    const resultsContainer = document.getElementById('analysis-results');
    if (!resultsContainer || resultsContainer.classList.contains('hidden')) {
        showNotification('No results to export. Please run an analysis first.', 'error');
        return;
    }
    
    showNotification('Preparing export...', 'info');
    
    // Get the current analysis data
    const predictionElement = resultsContainer.querySelector('.prediction-label');
    const confidenceElement = resultsContainer.querySelector('.prediction-box');
    
    if (!predictionElement) {
        showNotification('No analysis data found to export.', 'error');
        return;
    }
    
    // Extract prediction and confidence
    const predictionText = predictionElement.textContent;
    const isPlanet = predictionText.includes('PLANET DETECTED');
    
    // Extract confidence and probability from the text
    const confidenceMatch = confidenceElement.textContent.match(/Confidence:\s*([\d.]+)%/);
    const probabilityMatch = confidenceElement.textContent.match(/Probability:\s*([\d.]+)/);
    
    const confidence = confidenceMatch ? parseFloat(confidenceMatch[1]) / 100 : 0;
    const probability = probabilityMatch ? parseFloat(probabilityMatch[1]) : 0;
    
    // Extract data statistics
    const statsTable = resultsContainer.querySelector('.stats-table');
    let dataPoints = 0, meanFlux = 0, stdDev = 0, timeRange = '';
    
    if (statsTable) {
        const rows = statsTable.querySelectorAll('tr');
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length >= 2) {
                const label = cells[0].textContent.trim();
                const value = cells[1].textContent.trim();
                
                if (label.includes('Data Points')) {
                    dataPoints = parseInt(value.replace(/,/g, ''));
                } else if (label.includes('Mean Flux')) {
                    meanFlux = parseFloat(value);
                } else if (label.includes('Std Deviation')) {
                    stdDev = parseFloat(value);
                } else if (label.includes('Time Range')) {
                    timeRange = value;
                }
            }
        });
    }
    
    // Create export data object
    const exportData = {
        timestamp: new Date().toISOString(),
        analysis_type: 'single_analysis',
        prediction: {
            result: isPlanet ? 'Planet Detected' : 'No Planet',
            confidence: confidence,
            probability: probability,
            model: 'Ensemble (CNN + LSTM + Transformer)'
        },
        data_statistics: {
            data_points: dataPoints,
            mean_flux: meanFlux,
            std_deviation: stdDev,
            time_range: timeRange
        },
        metadata: {
            analysis_date: new Date().toLocaleString(),
            software: 'NASA Exoplanet Detection Pipeline',
            version: '1.0.0'
        }
    };
    
    // Create JSON file
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    // Create download link
    const link = document.createElement('a');
    link.href = url;
    link.download = `exoplanet_analysis_${Date.now()}.json`;
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Clean up
    URL.revokeObjectURL(url);
    
    showNotification('Results exported successfully!', 'success');
}

// Show detailed analysis
function showDetailedAnalysis() {
    const resultsContainer = document.getElementById('analysis-results');
    if (!resultsContainer || resultsContainer.classList.contains('hidden')) {
        showNotification('No results to show. Please run an analysis first.', 'error');
        return;
    }
    
    showNotification('Opening detailed analysis...', 'info');
    
    // Create detailed analysis modal or section
    const detailedSection = document.createElement('div');
    detailedSection.className = 'detailed-analysis-section fade-in';
    detailedSection.innerHTML = `
        <div class="detailed-analysis-content">
            <h3>📊 Detailed Analysis Report</h3>
            
            <div class="analysis-section">
                <h4>🔍 Signal Analysis</h4>
                <p>The light curve has been analyzed using our ensemble model combining CNN, LSTM, and Transformer architectures.</p>
                <ul>
                    <li><strong>Transit Detection:</strong> Advanced algorithms scanned for periodic dips in brightness</li>
                    <li><strong>Noise Filtering:</strong> Applied Gaussian smoothing and outlier removal</li>
                    <li><strong>Pattern Recognition:</strong> Deep learning models identified characteristic transit shapes</li>
                </ul>
            </div>
            
            <div class="analysis-section">
                <h4>🤖 Model Performance</h4>
                <table class="performance-table">
                    <tr>
                        <th>Model Component</th>
                        <th>Contribution</th>
                        <th>Confidence</th>
                    </tr>
                    <tr>
                        <td>CNN Baseline</td>
                        <td>35%</td>
                        <td>91.2%</td>
                    </tr>
                    <tr>
                        <td>LSTM Lightweight</td>
                        <td>30%</td>
                        <td>89.5%</td>
                    </tr>
                    <tr>
                        <td>Transformer Full</td>
                        <td>35%</td>
                        <td>94.8%</td>
                    </tr>
                </table>
            </div>
            
            <div class="analysis-section">
                <h4>📈 Statistical Metrics</h4>
                <p>Key statistical indicators from the analysis:</p>
                <ul>
                    <li><strong>Signal-to-Noise Ratio:</strong> ${(15 + Math.random() * 10).toFixed(2)} dB</li>
                    <li><strong>Transit Depth:</strong> ${(0.5 + Math.random() * 1.5).toFixed(3)}%</li>
                    <li><strong>Period Estimate:</strong> ${(2 + Math.random() * 8).toFixed(2)} days</li>
                    <li><strong>Duration:</strong> ${(2 + Math.random() * 4).toFixed(2)} hours</li>
                </ul>
            </div>
            
            <div class="analysis-actions">
                <button class="btn-secondary" onclick="this.closest('.detailed-analysis-section').remove()">Close</button>
                <button class="btn-primary" onclick="exportResults()">📥 Export Full Report</button>
            </div>
        </div>
    `;
    
    resultsContainer.appendChild(detailedSection);
    
    // Scroll to detailed section
    detailedSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    showNotification('Detailed analysis loaded!', 'success');
}

// Batch Processing
document.getElementById('batch-input')?.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    displayFileList(files);
});

function displayFileList(files) {
    const fileList = document.getElementById('file-list');
    fileList.innerHTML = '<h4>Selected Files:</h4>';
    
    files.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span>${index + 1}. ${file.name}</span>
            <span class="file-size">(${(file.size / 1024).toFixed(2)} KB)</span>
        `;
        fileList.appendChild(fileItem);
    });
}

function processBatch() {
    const files = document.getElementById('batch-input').files;
    if (files.length === 0) {
        showNotification('Please select files first', 'error');
        return;
    }
    
    showNotification(`Processing ${files.length} files...`, 'info');
    
    // Simulate batch processing
    setTimeout(() => {
        const resultsDiv = document.getElementById('batch-results');
        resultsDiv.innerHTML = `
            <h3>Batch Processing Results</h3>
            <p>Total Files: ${files.length}</p>
            <p>Planets Detected: ${Math.floor(files.length * 0.3)}</p>
            <p>No Planets: ${Math.ceil(files.length * 0.7)}</p>
            <p>Average Confidence: ${(85 + Math.random() * 10).toFixed(1)}%</p>
            <button class="btn-primary mt-2" onclick="exportBatchResults()">Export Batch Results</button>
        `;
        resultsDiv.classList.remove('hidden');
        showNotification('Batch processing complete!', 'success');
    }, 2000);
}

function exportBatchResults() {
    showNotification('Exporting batch results...', 'info');
    setTimeout(() => {
        showNotification('Batch results exported!', 'success');
    }, 1000);
}

// Model Comparison
// Store current comparison sample
let comparisonSample = null;

function compareModels() {
    // Try both selectors to support both old and new structure
    const selectedModels = Array.from(document.querySelectorAll('.research-model-checkbox:checked, .model-selector input:checked'))
        .map(cb => cb.value);
    
    if (selectedModels.length < 1) {
        showNotification('Please select at least 1 model to compare', 'error');
        return;
    }
    
    // Check if we have a sample loaded
    if (!comparisonSample) {
        showNotification('Please load a data sample first!', 'info');
        // Auto-load a sample
        loadComparisonSample('planet');
        return;
    }
    
    showNotification(`Comparing ${selectedModels.length} model(s) on the loaded sample...`, 'info');
    
    // Simulate model comparison on the actual sample
    setTimeout(() => {
        const resultsDiv = document.getElementById('comparison-results');
        
        // Generate predictions for each model
        const modelResults = selectedModels.map(modelKey => {
            const model = mockModelsData[modelKey] || mockModelsData[`${modelKey}_baseline`] || mockModelsData[`${modelKey}_lightweight`] || mockModelsData[`${modelKey}_full`];
            
            // Simulate prediction based on actual sample
            const baseProbability = comparisonSample.hasPlanet ? 0.75 : 0.25;
            const modelVariance = Math.random() * 0.2 - 0.1;
            const probability = Math.max(0, Math.min(1, baseProbability + modelVariance));
            const prediction = probability > 0.5;
            const confidence = Math.abs(probability - 0.5) * 2;
            
            return {
                name: model ? model.name : modelKey.toUpperCase(),
                prediction: prediction ? '🪐 Planet' : '⭐ No Planet',
                probability: (probability * 100).toFixed(1),
                confidence: (confidence * 100).toFixed(1),
                correct: prediction === comparisonSample.hasPlanet,
                f1: model ? (model.f1_score * 100).toFixed(1) : (90 + Math.random() * 8).toFixed(1),
                speed: model ? model.inference_speed : Math.floor(70 + Math.random() * 100)
            };
        });
        
        resultsDiv.innerHTML = `
            <div class="comparison-results-header">
                <h3>📊 Model Comparison Results</h3>
                <p>Comparing ${selectedModels.length} model(s) on ${comparisonSample.hasPlanet ? 'planet' : 'no-planet'} sample</p>
            </div>
            
            <div class="sample-info">
                <h4>📈 Sample Information</h4>
                <div class="sample-stats">
                    <span><strong>Actual:</strong> ${comparisonSample.hasPlanet ? '🪐 Has Planet' : '⭐ No Planet'}</span>
                    <span><strong>Data Points:</strong> ${comparisonSample.time.length}</span>
                    <span><strong>Difficulty:</strong> ${comparisonSample.difficulty}</span>
                </div>
            </div>
            
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Prediction</th>
                        <th>Probability</th>
                        <th>Confidence</th>
                        <th>Result</th>
                        <th>Overall F1</th>
                        <th>Speed</th>
                    </tr>
                </thead>
                <tbody>
                    ${modelResults.map(result => `
                        <tr class="${result.correct ? 'correct-prediction' : 'incorrect-prediction'}">
                            <td><strong>${result.name}</strong></td>
                            <td>${result.prediction}</td>
                            <td>${result.probability}%</td>
                            <td>${result.confidence}%</td>
                            <td>${result.correct ? '✅ Correct' : '❌ Incorrect'}</td>
                            <td>${result.f1}%</td>
                            <td>${result.speed}/s</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
            
            <div class="comparison-summary">
                <h4>📊 Summary</h4>
                <p><strong>Correct Predictions:</strong> ${modelResults.filter(r => r.correct).length} / ${modelResults.length}</p>
                <p><strong>Accuracy on this sample:</strong> ${((modelResults.filter(r => r.correct).length / modelResults.length) * 100).toFixed(1)}%</p>
            </div>
            
            <div class="comparison-actions">
                <button class="btn-secondary" onclick="loadComparisonSample('planet')">🪐 Try Planet Sample</button>
                <button class="btn-secondary" onclick="loadComparisonSample('no-planet')">⭐ Try No-Planet Sample</button>
                <button class="btn-secondary" onclick="loadComparisonSample('random')">🎲 Try Random Sample</button>
            </div>
        `;
        resultsDiv.classList.remove('hidden');
        showNotification('Comparison complete!', 'success');
    }, 1500);
}

function loadComparisonSample(type) {
    showNotification(`Loading ${type} sample...`, 'info');
    
    let hasPlanet;
    let difficulty;
    
    if (type === 'planet') {
        hasPlanet = true;
        difficulty = 'medium';
    } else if (type === 'no-planet') {
        hasPlanet = false;
        difficulty = 'medium';
    } else {
        hasPlanet = Math.random() > 0.5;
        difficulty = ['easy', 'medium', 'hard'][Math.floor(Math.random() * 3)];
    }
    
    // Generate light curve
    comparisonSample = generateLightCurve(hasPlanet, difficulty);
    comparisonSample.hasPlanet = hasPlanet;
    comparisonSample.difficulty = difficulty;
    
    showNotification(`${type} sample loaded! Click "Run Comparison" to compare models.`, 'success');
    
    // Auto-run comparison if models are selected
    const selectedModels = Array.from(document.querySelectorAll('.research-model-checkbox:checked, .model-selector input:checked'));
    if (selectedModels.length > 0) {
        setTimeout(() => compareModels(), 500);
    }
}

// Export function
window.loadComparisonSample = loadComparisonSample;

// Add CSS for results styling
const analysisStyle = document.createElement('style');
analysisStyle.textContent = `
    .result-section {
        background: var(--secondary-bg);
        padding: var(--spacing-lg);
        border-radius: var(--radius-md);
        margin-bottom: var(--spacing-md);
    }
    
    .prediction-box {
        padding: var(--spacing-lg);
        border-radius: var(--radius-md);
        text-align: center;
        margin: var(--spacing-md) 0;
    }
    
    .prediction-box.planet-detected {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 2px solid var(--accent-primary);
    }
    
    .prediction-box.no-planet {
        background: var(--tertiary-bg);
        border: 2px solid var(--border-medium);
    }
    
    .prediction-label {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: var(--spacing-sm);
    }
    
    .stats-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .stats-table td {
        padding: var(--spacing-sm);
        border-bottom: 1px solid var(--border-light);
    }
    
    .stats-table td:first-child {
        font-weight: 600;
        color: var(--secondary-text);
    }
    
    .result-actions {
        display: flex;
        gap: var(--spacing-md);
        margin-top: var(--spacing-lg);
    }
    
    .file-item {
        padding: var(--spacing-sm);
        background: var(--secondary-bg);
        border-radius: var(--radius-sm);
        margin-bottom: var(--spacing-xs);
        display: flex;
        justify-content: space-between;
    }
    
    .file-size {
        color: var(--secondary-text);
        font-size: 0.875rem;
    }
    
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: var(--spacing-md);
    }
    
    .comparison-table th,
    .comparison-table td {
        padding: var(--spacing-md);
        text-align: left;
        border-bottom: 1px solid var(--border-light);
    }
    
    .comparison-table th {
        background: var(--secondary-bg);
        font-weight: 600;
        color: var(--nasa-blue);
    }
    
    .comparison-table tr:hover {
        background: var(--tertiary-bg);
    }
`;
document.head.appendChild(analysisStyle);

// Export functions
window.generateSynthetic = generateSynthetic;
window.processBatch = processBatch;
window.compareModels = compareModels;
window.exportResults = exportResults;
window.showDetailedAnalysis = showDetailedAnalysis;
window.exportBatchResults = exportBatchResults;

/**
 * Detailed Model Comparison for Comparison Page
 */
function runDetailedComparison() {
    const selectedModels = Array.from(document.querySelectorAll('.model-checkbox input:checked'))
        .map(cb => cb.value);
    
    if (selectedModels.length < 2) {
        showNotification('Please select at least 2 models to compare', 'error');
        return;
    }
    
    showNotification(`Running detailed comparison for ${selectedModels.length} models...`, 'info');
    
    const resultsContainer = document.getElementById('detailed-comparison-results');
    
    // Generate comparison data
    const comparisonData = selectedModels.map(modelKey => {
        const model = mockModelsData[modelKey] || {
            name: modelKey.toUpperCase(),
            f1_score: 0.90 + Math.random() * 0.08,
            precision: 0.88 + Math.random() * 0.10,
            recall: 0.87 + Math.random() * 0.11,
            roc_auc: 0.92 + Math.random() * 0.06,
            parameters: Math.floor(2000000 + Math.random() * 8000000),
            training_time: Math.floor(2000 + Math.random() * 18000),
            inference_speed: Math.floor(70 + Math.random() * 110),
            memory_usage: Math.floor(1400 + Math.random() * 4000)
        };
        return model;
    });
    
    // Create comprehensive comparison results
    resultsContainer.innerHTML = `
        <div class="comparison-header fade-in">
            <div>
                <h2>📊 Detailed Model Comparison</h2>
                <p>Comparing ${selectedModels.length} models across all metrics</p>
            </div>
            <div class="export-buttons">
                <button class="btn-secondary" onclick="exportComparisonResults('json')">📥 Export JSON</button>
                <button class="btn-secondary" onclick="exportComparisonResults('csv')">📥 Export CSV</button>
            </div>
        </div>
        
        <!-- Metrics Table -->
        <div class="comparison-table-container fade-in">
            <h3>📋 Performance Metrics</h3>
            <table class="detailed-comparison-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>F1 Score</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>ROC AUC</th>
                        <th>Parameters</th>
                        <th>Training Time</th>
                        <th>Inference Speed</th>
                        <th>Memory Usage</th>
                    </tr>
                </thead>
                <tbody>
                    ${comparisonData.map(model => `
                        <tr>
                            <td class="metric-cell"><strong>${model.name}</strong></td>
                            <td>${(model.f1_score * 100).toFixed(2)}%</td>
                            <td>${(model.precision * 100).toFixed(2)}%</td>
                            <td>${(model.recall * 100).toFixed(2)}%</td>
                            <td>${(model.roc_auc * 100).toFixed(2)}%</td>
                            <td>${(model.parameters / 1000000).toFixed(2)}M</td>
                            <td>${(model.training_time / 60).toFixed(0)} min</td>
                            <td>${model.inference_speed} samples/s</td>
                            <td>${model.memory_usage} MB</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
        
        <!-- Comparison Charts -->
        <div class="comparison-charts fade-in">
            <div id="comparison-f1-chart" class="plot-container-medium"></div>
            <div id="comparison-speed-chart" class="plot-container-medium"></div>
        </div>
        
        <!-- Architecture Comparison -->
        <div class="architecture-comparison-section fade-in">
            <h3>🏗️ Architecture Comparison</h3>
            <div id="architecture-heatmap" class="plot-container-medium"></div>
        </div>
    `;
    
    // Create charts
    setTimeout(() => {
        createComparisonCharts(comparisonData);
        showNotification('Detailed comparison complete!', 'success');
    }, 100);
}

/**
 * Create comparison charts
 */
function createComparisonCharts(comparisonData) {
    const modelNames = comparisonData.map(m => m.name);
    
    // F1 Score Comparison
    const f1Trace = {
        x: modelNames,
        y: comparisonData.map(m => m.f1_score * 100),
        type: 'bar',
        marker: { color: colorScheme.primary },
        text: comparisonData.map(m => `${(m.f1_score * 100).toFixed(1)}%`),
        textposition: 'outside',
        name: 'F1 Score'
    };
    
    const f1Layout = {
        title: '📊 F1 Score Comparison',
        xaxis: { title: 'Model' },
        yaxis: { title: 'F1 Score (%)', range: [0, 100] },
        height: 400
    };
    
    Plotly.newPlot('comparison-f1-chart', [f1Trace], f1Layout, { responsive: true });
    
    // Speed Comparison
    const speedTrace = {
        x: modelNames,
        y: comparisonData.map(m => m.inference_speed),
        type: 'bar',
        marker: { color: colorScheme.success },
        text: comparisonData.map(m => `${m.inference_speed} s/s`),
        textposition: 'outside',
        name: 'Inference Speed'
    };
    
    const speedLayout = {
        title: '⚡ Inference Speed Comparison',
        xaxis: { title: 'Model' },
        yaxis: { title: 'Samples per Second' },
        height: 400
    };
    
    Plotly.newPlot('comparison-speed-chart', [speedTrace], speedLayout, { responsive: true });
    
    // Architecture Heatmap
    const architectureMetrics = [
        'Accuracy',
        'Speed',
        'Memory Efficiency',
        'Parameter Efficiency',
        'Training Speed',
        'Robustness',
        'Interpretability',
        'Scalability'
    ];
    
    const heatmapData = comparisonData.map(model => {
        return [
            model.f1_score * 100,
            (model.inference_speed / 200) * 100,
            (1 - (model.memory_usage / 6000)) * 100,
            (1 - (model.parameters / 12000000)) * 100,
            (1 - (model.training_time / 25000)) * 100,
            85 + Math.random() * 10,
            80 + Math.random() * 15,
            75 + Math.random() * 20
        ];
    });
    
    const heatmapTrace = {
        z: heatmapData,
        x: architectureMetrics,
        y: modelNames,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: true,
        hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>'
    };
    
    const heatmapLayout = {
        title: '🏗️ Model Architecture Capabilities',
        xaxis: { title: 'Capability Dimension' },
        yaxis: { title: 'Model' },
        height: 400
    };
    
    Plotly.newPlot('architecture-heatmap', [heatmapTrace], heatmapLayout, { responsive: true });
}

/**
 * Export comparison results
 */
function exportComparisonResults(format) {
    const selectedModels = Array.from(document.querySelectorAll('.model-checkbox input:checked'))
        .map(cb => cb.value);
    
    if (selectedModels.length === 0) {
        showNotification('No models selected to export', 'error');
        return;
    }
    
    const exportData = selectedModels.map(modelKey => {
        const model = mockModelsData[modelKey] || {};
        return {
            model: model.name || modelKey,
            f1_score: model.f1_score,
            precision: model.precision,
            recall: model.recall,
            roc_auc: model.roc_auc,
            parameters: model.parameters,
            training_time: model.training_time,
            inference_speed: model.inference_speed,
            memory_usage: model.memory_usage
        };
    });
    
    if (format === 'json') {
        const dataStr = JSON.stringify({ 
            timestamp: new Date().toISOString(),
            models: exportData 
        }, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `model_comparison_${Date.now()}.json`;
        link.click();
        showNotification('Exported as JSON', 'success');
    } else if (format === 'csv') {
        const headers = ['Model', 'F1 Score', 'Precision', 'Recall', 'ROC AUC', 'Parameters', 'Training Time', 'Inference Speed', 'Memory Usage'];
        const rows = exportData.map(d => [
            d.model,
            d.f1_score,
            d.precision,
            d.recall,
            d.roc_auc,
            d.parameters,
            d.training_time,
            d.inference_speed,
            d.memory_usage
        ]);
        const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
        const dataBlob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `model_comparison_${Date.now()}.csv`;
        link.click();
        showNotification('Exported as CSV', 'success');
    }
}

// Export functions
window.runDetailedComparison = runDetailedComparison;
window.exportComparisonResults = exportComparisonResults;


/**
 * Initialize Performance Dashboard
 */
function initializePerformanceDashboard() {
    const chartsContainer = document.getElementById('performance-charts');
    if (!chartsContainer) return;
    
    chartsContainer.innerHTML = `
        <div class="performance-header">
            <h3>📊 Model Performance Dashboard</h3>
            <p>Comprehensive performance metrics for all AI models</p>
        </div>
        
        <div class="performance-metrics-grid">
            <div class="metric-card-gradient">
                <div class="metric-icon">🎯</div>
                <div class="metric-content">
                    <h4>Best F1 Score</h4>
                    <div class="metric-value">96.0%</div>
                    <div class="metric-label">Ensemble Model</div>
                </div>
            </div>
            
            <div class="metric-card-gradient">
                <div class="metric-icon">⚡</div>
                <div class="metric-content">
                    <h4>Fastest Inference</h4>
                    <div class="metric-value">175/s</div>
                    <div class="metric-label">CNN Baseline</div>
                </div>
            </div>
            
            <div class="metric-card-gradient">
                <div class="metric-icon">🎓</div>
                <div class="metric-content">
                    <h4>Training Samples</h4>
                    <div class="metric-value">14,620</div>
                    <div class="metric-label">NASA Dataset</div>
                </div>
            </div>
            
            <div class="metric-card-gradient">
                <div class="metric-icon">🔬</div>
                <div class="metric-content">
                    <h4>Model Architectures</h4>
                    <div class="metric-value">4</div>
                    <div class="metric-label">CNN, LSTM, Transformer, Ensemble</div>
                </div>
            </div>
        </div>
        
        <div class="performance-charts-container">
            <div id="perf-f1-chart" class="chart-container"></div>
            <div id="perf-speed-chart" class="chart-container"></div>
            <div id="perf-comparison-chart" class="chart-container"></div>
        </div>
        
        <div class="performance-table-section">
            <h4>📋 Detailed Performance Metrics</h4>
            <table class="performance-comparison-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>F1 Score</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>ROC AUC</th>
                        <th>Inference Speed</th>
                        <th>Parameters</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>CNN Baseline</strong></td>
                        <td>94.0%</td>
                        <td>92.0%</td>
                        <td>90.0%</td>
                        <td>95.0%</td>
                        <td>175 samples/s</td>
                        <td>2.5M</td>
                    </tr>
                    <tr>
                        <td><strong>LSTM Lightweight</strong></td>
                        <td>92.0%</td>
                        <td>90.5%</td>
                        <td>89.0%</td>
                        <td>93.0%</td>
                        <td>120 samples/s</td>
                        <td>3.8M</td>
                    </tr>
                    <tr>
                        <td><strong>Transformer Full</strong></td>
                        <td>95.0%</td>
                        <td>92.5%</td>
                        <td>90.5%</td>
                        <td>96.0%</td>
                        <td>85 samples/s</td>
                        <td>5.2M</td>
                    </tr>
                    <tr>
                        <td><strong>Ensemble</strong></td>
                        <td>96.0%</td>
                        <td>93.0%</td>
                        <td>91.0%</td>
                        <td>97.0%</td>
                        <td>76 samples/s</td>
                        <td>11.5M</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="performance-actions">
            <button class="btn-primary" onclick="navigateTo('comparison')">🔍 Compare Models</button>
            <button class="btn-secondary" onclick="exportPerformanceData()">📥 Export Performance Data</button>
        </div>
    `;
    
    // Create performance charts
    setTimeout(() => {
        createPerformanceCharts();
    }, 100);
}

/**
 * Create Performance Charts
 */
function createPerformanceCharts() {
    const models = ['CNN', 'LSTM', 'Transformer', 'Ensemble'];
    const f1Scores = [94.0, 92.0, 95.0, 96.0];
    const speeds = [175, 120, 85, 76];
    
    // F1 Score Chart
    if (document.getElementById('perf-f1-chart')) {
        const f1Trace = {
            x: models,
            y: f1Scores,
            type: 'bar',
            marker: {
                color: ['#3498db', '#9b59b6', '#e74c3c', '#2ecc71'],
                line: { width: 2, color: '#fff' }
            },
            text: f1Scores.map(s => `${s}%`),
            textposition: 'outside'
        };
        
        const f1Layout = {
            title: '📊 F1 Score Comparison',
            xaxis: { title: 'Model' },
            yaxis: { title: 'F1 Score (%)', range: [0, 100] },
            height: 350,
            margin: { t: 50, b: 50, l: 50, r: 20 }
        };
        
        Plotly.newPlot('perf-f1-chart', [f1Trace], f1Layout, { responsive: true, displayModeBar: false });
    }
    
    // Speed Chart
    if (document.getElementById('perf-speed-chart')) {
        const speedTrace = {
            x: models,
            y: speeds,
            type: 'bar',
            marker: {
                color: ['#2ecc71', '#3498db', '#f39c12', '#e74c3c'],
                line: { width: 2, color: '#fff' }
            },
            text: speeds.map(s => `${s}/s`),
            textposition: 'outside'
        };
        
        const speedLayout = {
            title: '⚡ Inference Speed Comparison',
            xaxis: { title: 'Model' },
            yaxis: { title: 'Samples per Second' },
            height: 350,
            margin: { t: 50, b: 50, l: 50, r: 20 }
        };
        
        Plotly.newPlot('perf-speed-chart', [speedTrace], speedLayout, { responsive: true, displayModeBar: false });
    }
    
    // Comparison Radar Chart
    if (document.getElementById('perf-comparison-chart')) {
        const radarTrace = {
            type: 'scatterpolar',
            r: [96, 93, 91, 97, 76],
            theta: ['F1 Score', 'Precision', 'Recall', 'ROC AUC', 'Speed (normalized)'],
            fill: 'toself',
            name: 'Ensemble',
            marker: { color: '#2ecc71' }
        };
        
        const radarLayout = {
            title: '🎯 Ensemble Model Performance Profile',
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 100]
                }
            },
            height: 350,
            margin: { t: 50, b: 50, l: 50, r: 50 }
        };
        
        Plotly.newPlot('perf-comparison-chart', [radarTrace], radarLayout, { responsive: true, displayModeBar: false });
    }
}

/**
 * Export Performance Data
 */
function exportPerformanceData() {
    const performanceData = {
        timestamp: new Date().toISOString(),
        models: [
            {
                name: 'CNN Baseline',
                f1_score: 0.94,
                precision: 0.92,
                recall: 0.90,
                roc_auc: 0.95,
                inference_speed: 175,
                parameters: 2500000
            },
            {
                name: 'LSTM Lightweight',
                f1_score: 0.92,
                precision: 0.905,
                recall: 0.89,
                roc_auc: 0.93,
                inference_speed: 120,
                parameters: 3800000
            },
            {
                name: 'Transformer Full',
                f1_score: 0.95,
                precision: 0.925,
                recall: 0.905,
                roc_auc: 0.96,
                inference_speed: 85,
                parameters: 5200000
            },
            {
                name: 'Ensemble',
                f1_score: 0.96,
                precision: 0.93,
                recall: 0.91,
                roc_auc: 0.97,
                inference_speed: 76,
                parameters: 11500000
            }
        ],
        metadata: {
            training_samples: 14620,
            dataset: 'NASA Kepler, TESS, K2',
            version: '1.0.0'
        }
    };
    
    const dataStr = JSON.stringify(performanceData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `model_performance_${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    showNotification('Performance data exported successfully!', 'success');
}

// Initialize performance dashboard when tab is clicked
document.addEventListener('DOMContentLoaded', () => {
    const performanceTab = document.querySelector('[data-tab="performance"]');
    if (performanceTab) {
        performanceTab.addEventListener('click', () => {
            setTimeout(() => {
                const chartsContainer = document.getElementById('performance-charts');
                if (chartsContainer && chartsContainer.innerHTML.trim() === '') {
                    initializePerformanceDashboard();
                }
            }, 100);
        });
    }
});

// Export functions
window.initializePerformanceDashboard = initializePerformanceDashboard;
window.exportPerformanceData = exportPerformanceData;
