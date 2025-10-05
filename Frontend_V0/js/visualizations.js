// NASA Exoplanet Detection Pipeline - Advanced Visualizations
// Port of the Plotly-based interactive visualizations to the frontend

// Color Scheme
const colorScheme = {
    primary: '#1f77b4',
    secondary: '#ff7f0e',
    success: '#2ecc71',
    danger: '#e74c3c',
    warning: '#f39c12',
    info: '#3498db',
    planet: '#9b59b6',
    noPlanet: '#95a5a6'
};

// Mock Model Data (replace with real data from backend)
const mockModelsData = {
    cnn_baseline: {
        name: 'CNN Baseline',
        f1_score: 0.94,
        precision: 0.92,
        recall: 0.90,
        roc_auc: 0.95,
        parameters: 2500000,
        training_time: 2700,
        inference_speed: 175,
        memory_usage: 1800
    },
    lstm_lightweight: {
        name: 'LSTM Lightweight',
        f1_score: 0.92,
        precision: 0.905,
        recall: 0.89,
        roc_auc: 0.93,
        parameters: 3800000,
        training_time: 7200,
        inference_speed: 120,
        memory_usage: 2200
    },
    transformer_full: {
        name: 'Transformer Full',
        f1_score: 0.95,
        precision: 0.925,
        recall: 0.905,
        roc_auc: 0.96,
        parameters: 5200000,
        training_time: 10800,
        inference_speed: 85,
        memory_usage: 1428
    },
    ensemble: {
        name: 'Ensemble',
        f1_score: 0.975,
        precision: 0.962,
        recall: 0.988,
        roc_auc: 0.991,
        parameters: 11500000,
        training_time: 20700,
        inference_speed: 76,
        memory_usage: 5428
    }
};

/**
 * Create Interactive Performance Dashboard (9-panel)
 */
function createPerformanceDashboard(containerId) {
    const models = Object.values(mockModelsData);
    const modelNames = models.map(m => m.name);

    // Create subplots
    const traces = [];

    // 1. F1 Score Comparison
    traces.push({
        x: modelNames,
        y: models.map(m => m.f1_score * 100),
        type: 'bar',
        name: 'F1 Score',
        marker: { color: colorScheme.primary },
        text: models.map(m => `${(m.f1_score * 100).toFixed(1)}%`),
        textposition: 'outside',
        xaxis: 'x',
        yaxis: 'y'
    });

    // 2. Precision vs Recall
    traces.push({
        x: models.map(m => m.precision * 100),
        y: models.map(m => m.recall * 100),
        mode: 'markers+text',
        type: 'scatter',
        name: 'Precision vs Recall',
        text: modelNames,
        textposition: 'top center',
        marker: {
            size: 15,
            color: models.map(m => m.f1_score),
            colorscale: 'Viridis',
            showscale: true,
            colorbar: { title: 'F1 Score', x: 1.15 }
        },
        xaxis: 'x2',
        yaxis: 'y2'
    });

    // 3. ROC-AUC Scores
    traces.push({
        x: modelNames,
        y: models.map(m => m.roc_auc * 100),
        type: 'bar',
        name: 'ROC-AUC',
        marker: { color: colorScheme.success },
        text: models.map(m => `${(m.roc_auc * 100).toFixed(1)}%`),
        textposition: 'outside',
        xaxis: 'x3',
        yaxis: 'y3'
    });

    // 4. Training Time
    traces.push({
        x: modelNames,
        y: models.map(m => m.training_time / 60),
        type: 'bar',
        name: 'Training Time',
        marker: { color: colorScheme.warning },
        text: models.map(m => `${(m.training_time / 60).toFixed(1)}m`),
        textposition: 'outside',
        xaxis: 'x4',
        yaxis: 'y4'
    });

    // 5. Inference Speed
    traces.push({
        x: modelNames,
        y: models.map(m => m.inference_speed),
        type: 'bar',
        name: 'Inference Speed',
        marker: { color: colorScheme.info },
        text: models.map(m => `${m.inference_speed}/s`),
        textposition: 'outside',
        xaxis: 'x5',
        yaxis: 'y5'
    });

    // 6. Memory vs Parameters
    traces.push({
        x: models.map(m => m.parameters / 1e6),
        y: models.map(m => m.memory_usage),
        mode: 'markers+text',
        type: 'scatter',
        name: 'Memory Usage',
        text: modelNames,
        textposition: 'top center',
        marker: { size: 12, color: colorScheme.danger },
        xaxis: 'x6',
        yaxis: 'y6'
    });

    // 7. Parameter Efficiency
    const paramEfficiency = models.map(m => m.f1_score / (m.parameters / 1e6));
    traces.push({
        x: modelNames,
        y: paramEfficiency,
        type: 'bar',
        name: 'Param Efficiency',
        marker: { color: colorScheme.planet },
        text: paramEfficiency.map(e => e.toFixed(2)),
        textposition: 'outside',
        xaxis: 'x7',
        yaxis: 'y7'
    });

    // 8. F1 Score Distribution (box plot)
    models.forEach((model, idx) => {
        traces.push({
            y: [model.f1_score * 100],
            type: 'box',
            name: model.name,
            boxmean: 'sd',
            marker: { color: colorScheme.primary },
            xaxis: 'x8',
            yaxis: 'y8',
            showlegend: false
        });
    });

    // 9. Radar Chart (top 4 models)
    const topModels = models.slice(0, 4);
    const radarColors = [colorScheme.danger, colorScheme.warning, colorScheme.success, colorScheme.planet];

    topModels.forEach((model, idx) => {
        traces.push({
            type: 'scatterpolar',
            r: [
                model.f1_score * 100,
                model.precision * 100,
                model.recall * 100,
                model.roc_auc * 100,
                model.f1_score * 100
            ],
            theta: ['F1 Score', 'Precision', 'Recall', 'ROC-AUC', 'F1 Score'],
            fill: 'toself',
            name: model.name,
            line: { color: radarColors[idx] },
            opacity: 0.6,
            subplot: 'polar'
        });
    });

    const layout = {
        title: {
            text: '🤖 Comprehensive Model Performance Dashboard',
            font: { size: 20 }
        },
        grid: {
            rows: 3,
            columns: 3,
            pattern: 'independent',
            roworder: 'top to bottom'
        },
        xaxis: { title: 'Model', domain: [0, 0.3] },
        yaxis: { title: 'F1 Score (%)', domain: [0.7, 1] },

        xaxis2: { title: 'Precision (%)', domain: [0.35, 0.65] },
        yaxis2: { title: 'Recall (%)', domain: [0.7, 1] },

        xaxis3: { title: 'Model', domain: [0.7, 1] },
        yaxis3: { title: 'ROC-AUC (%)', domain: [0.7, 1] },

        xaxis4: { title: 'Model', domain: [0, 0.3] },
        yaxis4: { title: 'Time (min)', domain: [0.35, 0.65] },

        xaxis5: { title: 'Model', domain: [0.35, 0.65] },
        yaxis5: { title: 'Samples/s', domain: [0.35, 0.65] },

        xaxis6: { title: 'Parameters (M)', domain: [0.7, 1] },
        yaxis6: { title: 'Memory (MB)', domain: [0.35, 0.65] },

        xaxis7: { title: 'Model', domain: [0, 0.3] },
        yaxis7: { title: 'F1/M Params', domain: [0, 0.3] },

        xaxis8: { domain: [0.35, 0.65] },
        yaxis8: { title: 'F1 Score (%)', domain: [0, 0.3] },

        polar: {
            domain: { x: [0.7, 1], y: [0, 0.3] },
            radialaxis: { visible: true, range: [80, 100] }
        },

        height: 1200,
        showlegend: true,
        hovermode: 'closest'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };

    Plotly.newPlot(containerId, traces, layout, config);
}

/**
 * Create Interactive Light Curve Analyzer
 */
function createLightCurveAnalyzer(containerId, time, flux, prediction = null) {
    const traces = [];

    // Main light curve
    traces.push({
        x: time,
        y: flux,
        mode: 'lines',
        name: 'Observed Flux',
        line: { color: colorScheme.primary, width: 1.5 },
        hovertemplate: 'Time: %{x:.3f} days<br>Flux: %{y:.6f}<extra></extra>'
    });

    // Add smoothed trend
    const windowSize = Math.max(10, Math.floor(flux.length / 100));
    const fluxSmooth = movingAverage(flux, windowSize);

    traces.push({
        x: time,
        y: fluxSmooth,
        mode: 'lines',
        name: 'Smoothed Trend',
        line: { color: colorScheme.secondary, width: 2, dash: 'dash' },
        opacity: 0.7
    });

    // Detect and mark potential transits
    const fluxMedian = median(flux);
    const fluxStd = standardDeviation(flux);
    const transitThreshold = fluxMedian - 2 * fluxStd;

    const transitIndices = [];
    const transitTimes = [];
    const transitFluxes = [];

    flux.forEach((f, i) => {
        if (f < transitThreshold) {
            transitIndices.push(i);
            transitTimes.push(time[i]);
            transitFluxes.push(f);
        }
    });

    if (transitTimes.length > 0) {
        traces.push({
            x: transitTimes,
            y: transitFluxes,
            mode: 'markers',
            name: 'Potential Transits',
            marker: {
                color: colorScheme.danger,
                size: 6,
                symbol: 'diamond'
            },
            hovertemplate: 'Transit at: %{x:.3f} days<br>Depth: %{y:.6f}<extra></extra>'
        });
    }

    const layout = {
        title: '🔍 Interactive Light Curve Analysis',
        xaxis: {
            title: 'Time (days)',
            rangeslider: { visible: true, thickness: 0.05 },
            rangeselector: {
                buttons: [
                    { count: 1, label: '1d', step: 'day', stepmode: 'backward' },
                    { count: 7, label: '1w', step: 'day', stepmode: 'backward' },
                    { count: 14, label: '2w', step: 'day', stepmode: 'backward' },
                    { step: 'all', label: 'All' }
                ]
            }
        },
        yaxis: { title: 'Normalized Flux' },
        height: 500,
        hovermode: 'x unified',
        plot_bgcolor: '#ffffff',
        paper_bgcolor: '#ffffff'
    };

    // Add prediction annotation if available
    if (prediction) {
        layout.annotations = [{
            x: time[Math.floor(time.length / 4)],
            y: Math.max(...flux),
            text: `🤖 AI Prediction<br>Planet Probability: ${(prediction.probability * 100).toFixed(1)}%<br>Confidence: ${(prediction.confidence * 100).toFixed(1)}%`,
            showarrow: true,
            arrowhead: 2,
            arrowcolor: prediction.probability > 0.5 ? colorScheme.success : colorScheme.info,
            bgcolor: 'rgba(255,255,255,0.9)',
            bordercolor: prediction.probability > 0.5 ? colorScheme.success : colorScheme.info,
            borderwidth: 2
        }];
    }

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };

    Plotly.newPlot(containerId, traces, layout, config);
}

/**
 * Create Architecture Comparison Heatmap
 */
function createArchitectureComparison(containerId) {
    const features = [
        'Temporal Modeling',
        'Attention Mechanism',
        'Parallel Processing',
        'Memory Efficiency',
        'Training Speed',
        'Inference Speed',
        'Long-range Dependencies',
        'Variable Sequence'
    ];

    const models = ['CNN', 'LSTM', 'Transformer', 'Ensemble'];

    // Capability scores (0-1 scale)
    const scores = [
        [0.6, 0.2, 1.0, 1.0, 1.0, 1.0, 0.4, 0.3], // CNN
        [1.0, 0.8, 0.3, 0.7, 0.6, 0.6, 0.7, 1.0], // LSTM
        [1.0, 1.0, 1.0, 0.4, 0.4, 0.7, 1.0, 1.0], // Transformer
        [0.9, 0.9, 0.7, 0.5, 0.5, 0.6, 0.9, 0.9]  // Ensemble
    ];

    const trace = {
        z: scores,
        x: models,
        y: features,
        type: 'heatmap',
        colorscale: 'RdYlGn',
        text: scores.map(row => row.map(val => val.toFixed(1))),
        texttemplate: '%{text}',
        textfont: { size: 12 },
        colorbar: { title: 'Capability<br>Score' },
        hovertemplate: 'Model: %{x}<br>Feature: %{y}<br>Score: %{z:.2f}<extra></extra>'
    };

    const layout = {
        title: '🏗️ Model Architecture Feature Comparison',
        xaxis: { title: 'Model' },
        yaxis: { title: 'Capability' },
        height: 500,
        font: { size: 11 }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };

    Plotly.newPlot(containerId, [trace], layout, config);
}

/**
 * Create Efficiency Scatter Matrix
 */
function createEfficiencyScatterMatrix(containerId) {
    const models = Object.values(mockModelsData);
    const modelNames = models.map(m => m.name);

    const traces = [];

    // 1. Parameters vs F1
    traces.push({
        x: models.map(m => m.parameters / 1e6),
        y: models.map(m => m.f1_score * 100),
        mode: 'markers+text',
        type: 'scatter',
        text: modelNames,
        textposition: 'top center',
        marker: {
            size: 12,
            color: models.map(m => m.f1_score),
            colorscale: 'Viridis',
            showscale: false
        },
        name: 'Models',
        xaxis: 'x',
        yaxis: 'y'
    });

    // 2. Training Time vs F1
    traces.push({
        x: models.map(m => m.training_time / 60),
        y: models.map(m => m.f1_score * 100),
        mode: 'markers+text',
        type: 'scatter',
        text: modelNames,
        textposition: 'top center',
        marker: { size: 12, color: colorScheme.warning },
        name: 'Training',
        xaxis: 'x2',
        yaxis: 'y2'
    });

    // 3. Memory vs Inference Speed
    traces.push({
        x: models.map(m => m.memory_usage),
        y: models.map(m => m.inference_speed),
        mode: 'markers+text',
        type: 'scatter',
        text: modelNames,
        textposition: 'top center',
        marker: { size: 12, color: colorScheme.info },
        name: 'Efficiency',
        xaxis: 'x3',
        yaxis: 'y3'
    });

    // 4. Parameter Efficiency
    const paramEff = models.map(m => m.f1_score / (m.parameters / 1e6));
    traces.push({
        x: modelNames,
        y: paramEff,
        type: 'bar',
        marker: { color: colorScheme.success },
        text: paramEff.map(e => e.toFixed(2)),
        textposition: 'outside',
        name: 'Param Efficiency',
        xaxis: 'x4',
        yaxis: 'y4'
    });

    const layout = {
        title: '⚡ Comprehensive Efficiency Analysis',
        grid: { rows: 2, columns: 2, pattern: 'independent' },
        xaxis: { title: 'Parameters (M)', domain: [0, 0.45] },
        yaxis: { title: 'F1 Score (%)', domain: [0.55, 1] },
        xaxis2: { title: 'Training Time (min)', domain: [0.55, 1] },
        yaxis2: { title: 'F1 Score (%)', domain: [0.55, 1] },
        xaxis3: { title: 'Memory (MB)', domain: [0, 0.45] },
        yaxis3: { title: 'Inference Speed (samples/s)', domain: [0, 0.45] },
        xaxis4: { title: 'Model', domain: [0.55, 1] },
        yaxis4: { title: 'F1 / M Params', domain: [0, 0.45] },
        height: 700,
        showlegend: false
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };

    Plotly.newPlot(containerId, traces, layout, config);
}

/**
 * Create Metric Cards with Gradients
 */
function createMetricCard(title, value, delta, icon) {
    return `
        <div class="metric-card-gradient">
            <div class="metric-icon">${icon}</div>
            <div class="metric-content">
                <div class="metric-title">${title}</div>
                <div class="metric-value">${value}</div>
                ${delta ? `<div class="metric-delta">${delta}</div>` : ''}
            </div>
        </div>
    `;
}

// Utility Functions
function movingAverage(arr, windowSize) {
    const result = [];
    for (let i = 0; i < arr.length; i++) {
        const start = Math.max(0, i - Math.floor(windowSize / 2));
        const end = Math.min(arr.length, i + Math.floor(windowSize / 2) + 1);
        const window = arr.slice(start, end);
        const avg = window.reduce((a, b) => a + b, 0) / window.length;
        result.push(avg);
    }
    return result;
}

function median(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function standardDeviation(arr) {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const variance = arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length;
    return Math.sqrt(variance);
}

// Export functions
window.createPerformanceDashboard = createPerformanceDashboard;
window.createLightCurveAnalyzer = createLightCurveAnalyzer;
window.createArchitectureComparison = createArchitectureComparison;
window.createEfficiencyScatterMatrix = createEfficiencyScatterMatrix;
window.createMetricCard = createMetricCard;
window.mockModelsData = mockModelsData;


/**
 * Run Detailed Model Comparison
 * This is the full implementation of the "coming soon" feature
 */
function runDetailedComparison() {
    const selectedModels = Array.from(document.querySelectorAll('.model-checkboxes input:checked'))
        .map(cb => cb.value);

    if (selectedModels.length < 2) {
        showNotification('Please select at least 2 models to compare', 'error');
        return;
    }

    showNotification(`Comparing ${selectedModels.length} models...`, 'info');

    const resultsContainer = document.getElementById('detailed-comparison-results');

    // Create comprehensive comparison view
    resultsContainer.innerHTML = `
        <div class="comparison-header">
            <h3>🔍 Detailed Model Comparison Results</h3>
            <button class="btn-primary" onclick="exportComparisonResults()">📥 Export Results</button>
        </div>
        
        <!-- Performance Metrics Cards -->
        <div class="metric-cards-grid">
            ${createMetricCard('Models Compared', selectedModels.length.toString(), 'Selected', '🤖')}
            ${createMetricCard('Best F1 Score', '97.5%', 'Ensemble', '🏆')}
            ${createMetricCard('Fastest Model', 'CNN', '175/s', '⚡')}
            ${createMetricCard('Most Efficient', 'Transformer', '2.55', '🎯')}
        </div>
        
        <!-- Comparison Table -->
        <div class="comparison-table-container">
            <h4>📊 Performance Metrics Comparison</h4>
            <table class="detailed-comparison-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>F1 Score</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>ROC-AUC</th>
                        <th>Parameters</th>
                        <th>Speed</th>
                        <th>Memory</th>
                    </tr>
                </thead>
                <tbody id="comparison-table-body"></tbody>
            </table>
        </div>
        
        <!-- Interactive Charts -->
        <div class="comparison-charts">
            <div id="comparison-performance-chart" class="plot-container-medium"></div>
            <div id="comparison-efficiency-chart" class="plot-container-medium"></div>
        </div>
        
        <!-- Architecture Details -->
        <div class="architecture-comparison-section">
            <h4>🏗️ Architecture Comparison</h4>
            <div id="comparison-architecture-heatmap" class="plot-container-medium"></div>
        </div>
        
        <!-- Export Options -->
        <div class="export-panel">
            <h4>📥 Export Options</h4>
            <div class="export-buttons">
                <button class="btn-secondary" onclick="exportAsJSON()">📄 Export as JSON</button>
                <button class="btn-secondary" onclick="exportAsCSV()">📊 Export as CSV</button>
                <button class="btn-secondary" onclick="exportAsPDF()">📑 Export as PDF</button>
                <button class="btn-secondary" onclick="exportCharts()">🖼️ Export Charts</button>
            </div>
        </div>
    `;

    // Populate comparison table
    populateComparisonTable(selectedModels);

    // Create comparison charts
    setTimeout(() => {
        createComparisonPerformanceChart(selectedModels);
        createComparisonEfficiencyChart(selectedModels);
        createArchitectureComparison('comparison-architecture-heatmap');
    }, 100);

    showNotification('Comparison complete!', 'success');
}

/**
 * Populate Comparison Table
 */
function populateComparisonTable(selectedModels) {
    const tbody = document.getElementById('comparison-table-body');
    if (!tbody) return;

    tbody.innerHTML = '';

    selectedModels.forEach(modelKey => {
        const model = mockModelsData[modelKey];
        if (!model) return;

        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${model.name}</strong></td>
            <td class="metric-cell">${(model.f1_score * 100).toFixed(2)}%</td>
            <td class="metric-cell">${(model.precision * 100).toFixed(2)}%</td>
            <td class="metric-cell">${(model.recall * 100).toFixed(2)}%</td>
            <td class="metric-cell">${(model.roc_auc * 100).toFixed(2)}%</td>
            <td class="metric-cell">${(model.parameters / 1e6).toFixed(2)}M</td>
            <td class="metric-cell">${model.inference_speed}/s</td>
            <td class="metric-cell">${model.memory_usage}MB</td>
        `;
        tbody.appendChild(row);
    });
}

/**
 * Create Comparison Performance Chart
 */
function createComparisonPerformanceChart(selectedModels) {
    const models = selectedModels.map(key => mockModelsData[key]).filter(m => m);
    const modelNames = models.map(m => m.name);

    const traces = [
        {
            x: modelNames,
            y: models.map(m => m.f1_score * 100),
            name: 'F1 Score',
            type: 'bar',
            marker: { color: colorScheme.primary }
        },
        {
            x: modelNames,
            y: models.map(m => m.precision * 100),
            name: 'Precision',
            type: 'bar',
            marker: { color: colorScheme.success }
        },
        {
            x: modelNames,
            y: models.map(m => m.recall * 100),
            name: 'Recall',
            type: 'bar',
            marker: { color: colorScheme.warning }
        },
        {
            x: modelNames,
            y: models.map(m => m.roc_auc * 100),
            name: 'ROC-AUC',
            type: 'bar',
            marker: { color: colorScheme.info }
        }
    ];

    const layout = {
        title: '📊 Performance Metrics Comparison',
        barmode: 'group',
        xaxis: { title: 'Model' },
        yaxis: { title: 'Score (%)', range: [80, 100] },
        height: 400,
        hovermode: 'closest'
    };

    const config = { responsive: true, displayModeBar: true, displaylogo: false };

    Plotly.newPlot('comparison-performance-chart', traces, layout, config);
}

/**
 * Create Comparison Efficiency Chart
 */
function createComparisonEfficiencyChart(selectedModels) {
    const models = selectedModels.map(key => mockModelsData[key]).filter(m => m);
    const modelNames = models.map(m => m.name);

    const trace = {
        x: models.map(m => m.parameters / 1e6),
        y: models.map(m => m.f1_score * 100),
        mode: 'markers+text',
        type: 'scatter',
        text: modelNames,
        textposition: 'top center',
        marker: {
            size: models.map(m => m.inference_speed / 10),
            color: models.map(m => m.memory_usage),
            colorscale: 'Viridis',
            showscale: true,
            colorbar: { title: 'Memory<br>(MB)' }
        },
        hovertemplate: '<b>%{text}</b><br>Params: %{x:.2f}M<br>F1: %{y:.2f}%<extra></extra>'
    };

    const layout = {
        title: '⚡ Efficiency Analysis: Parameters vs Performance',
        xaxis: { title: 'Parameters (Millions)' },
        yaxis: { title: 'F1 Score (%)' },
        height: 400,
        hovermode: 'closest'
    };

    const config = { responsive: true, displayModeBar: true, displaylogo: false };

    Plotly.newPlot('comparison-efficiency-chart', [trace], layout, config);
}

/**
 * Export Functions
 */
function exportComparisonResults() {
    showNotification('Preparing export...', 'info');

    const exportData = {
        timestamp: new Date().toISOString(),
        models: Object.values(mockModelsData),
        summary: {
            total_models: Object.keys(mockModelsData).length,
            best_f1: Math.max(...Object.values(mockModelsData).map(m => m.f1_score)),
            best_model: Object.values(mockModelsData).reduce((a, b) => a.f1_score > b.f1_score ? a : b).name
        }
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `model_comparison_${Date.now()}.json`;
    link.click();

    showNotification('Results exported successfully!', 'success');
}

function exportAsJSON() {
    exportComparisonResults();
}

function exportAsCSV() {
    showNotification('Exporting as CSV...', 'info');

    const models = Object.values(mockModelsData);
    let csv = 'Model,F1 Score,Precision,Recall,ROC-AUC,Parameters,Inference Speed,Memory Usage\n';

    models.forEach(model => {
        csv += `${model.name},${model.f1_score},${model.precision},${model.recall},${model.roc_auc},${model.parameters},${model.inference_speed},${model.memory_usage}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `model_comparison_${Date.now()}.csv`;
    link.click();

    showNotification('CSV exported successfully!', 'success');
}

function exportAsPDF() {
    showNotification('PDF export feature coming soon!', 'info');
    // PDF export functionality to be implemented
}

function exportCharts() {
    showNotification('Exporting charts...', 'info');

    // Export all visible Plotly charts as PNG
    const charts = document.querySelectorAll('.plot-container-medium, .plot-container-large');
    let exportCount = 0;

    charts.forEach((chart, index) => {
        if (chart.data && chart.data.length > 0) {
            Plotly.downloadImage(chart, {
                format: 'png',
                width: 1200,
                height: 800,
                filename: `chart_${index + 1}_${Date.now()}`
            });
            exportCount++;
        }
    });

    if (exportCount > 0) {
        showNotification(`Exported ${exportCount} charts successfully!`, 'success');
    } else {
        showNotification('No charts to export', 'error');
    }
}

/**
 * Integrate Model Comparison into Research Mode
 */
function enhanceResearchModeWithComparison() {
    const compareTab = document.getElementById('compare-tab');
    if (!compareTab) return;

    // Replace the simple comparison with detailed version
    compareTab.innerHTML = `
        <div class="comparison-container-enhanced">
            <h3>🔍 Detailed Model Comparison</h3>
            <p class="section-description">Compare multiple models on the same light curve data</p>
            
            <!-- Model Selection -->
            <div class="model-selection-grid">
                <label class="model-card">
                    <input type="checkbox" value="cnn" checked class="research-model-checkbox" />
                    <div class="model-card-content">
                        <h4>🔷 CNN Baseline</h4>
                        <p>Fast & Efficient</p>
                        <div class="model-stats">
                            <span>F1: 94.0%</span>
                            <span>Speed: 175/s</span>
                        </div>
                    </div>
                </label>
                
                <label class="model-card">
                    <input type="checkbox" value="lstm" checked class="research-model-checkbox" />
                    <div class="model-card-content">
                        <h4>🔷 LSTM Lightweight</h4>
                        <p>Sequential Modeling</p>
                        <div class="model-stats">
                            <span>F1: 92.0%</span>
                            <span>Speed: 120/s</span>
                        </div>
                    </div>
                </label>
                
                <label class="model-card">
                    <input type="checkbox" value="transformer" checked class="research-model-checkbox" />
                    <div class="model-card-content">
                        <h4>🔷 Transformer Full</h4>
                        <p>State-of-the-Art</p>
                        <div class="model-stats">
                            <span>F1: 95.0%</span>
                            <span>Speed: 85/s</span>
                        </div>
                    </div>
                </label>
                
                <label class="model-card">
                    <input type="checkbox" value="ensemble" checked class="research-model-checkbox" />
                    <div class="model-card-content">
                        <h4>🔷 Ensemble</h4>
                        <p>Best Performance</p>
                        <div class="model-stats">
                            <span>F1: 96.0%</span>
                            <span>Speed: 76/s</span>
                        </div>
                    </div>
                </label>
            </div>
            
            <div class="comparison-actions">
                <button class="btn-primary" onclick="runResearchComparison()">🚀 Run Comparison</button>
                <button class="btn-secondary" onclick="loadComparisonExample()">🎲 Load Example Data</button>
            </div>
            
            <!-- Results Area -->
            <div id="research-comparison-results" class="research-comparison-results"></div>
        </div>
    `;
}

/**
 * Run Research Mode Comparison
 */
function runResearchComparison() {
    const selectedModels = Array.from(document.querySelectorAll('.research-model-checkbox:checked'))
        .map(cb => cb.value);

    if (selectedModels.length < 2) {
        showNotification('Please select at least 2 models', 'error');
        return;
    }

    showNotification('Running comparison on all selected models...', 'info');

    // Generate sample light curve
    const { time, flux } = generateLightCurve(true, 'medium');

    const resultsContainer = document.getElementById('research-comparison-results');

    // Create results display
    let resultsHTML = `
        <div class="comparison-results-header">
            <h4>📊 Comparison Results</h4>
            <button class="btn-primary" onclick="exportResearchResults()">📥 Export All Results</button>
        </div>
        
        <!-- Light Curve Plot -->
        <div id="research-comparison-lightcurve" class="plot-container-medium"></div>
        
        <!-- Model Predictions Table -->
        <div class="predictions-table-container">
            <h5>🤖 Model Predictions</h5>
            <table class="predictions-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Prediction</th>
                        <th>Probability</th>
                        <th>Confidence</th>
                        <th>Inference Time</th>
                    </tr>
                </thead>
                <tbody>
    `;

    selectedModels.forEach(modelKey => {
        const model = mockModelsData[modelKey];
        const probability = 0.7 + Math.random() * 0.25;
        const confidence = 0.8 + Math.random() * 0.15;
        const inferenceTime = (1000 / model.inference_speed).toFixed(2);
        const prediction = probability > 0.5 ? '🪐 Planet' : '⭐ No Planet';

        resultsHTML += `
            <tr>
                <td><strong>${model.name}</strong></td>
                <td class="${probability > 0.5 ? 'planet-pred' : 'no-planet-pred'}">${prediction}</td>
                <td>${(probability * 100).toFixed(1)}%</td>
                <td>${(confidence * 100).toFixed(1)}%</td>
                <td>${inferenceTime}ms</td>
            </tr>
        `;
    });

    resultsHTML += `
                </tbody>
            </table>
        </div>
        
        <!-- Comparison Charts -->
        <div class="comparison-charts-grid">
            <div id="research-comparison-bar" class="plot-container-small"></div>
            <div id="research-comparison-radar" class="plot-container-small"></div>
        </div>
    `;

    resultsContainer.innerHTML = resultsHTML;

    // Create visualizations
    setTimeout(() => {
        createLightCurveAnalyzer('research-comparison-lightcurve', time, flux, null);
        createComparisonBarChart(selectedModels, 'research-comparison-bar');
        createComparisonRadarChart(selectedModels, 'research-comparison-radar');
    }, 100);
}

/**
 * Create Comparison Bar Chart
 */
function createComparisonBarChart(selectedModels, containerId) {
    const models = selectedModels.map(key => mockModelsData[key]).filter(m => m);
    const modelNames = models.map(m => m.name);

    const trace = {
        x: modelNames,
        y: models.map(m => m.f1_score * 100),
        type: 'bar',
        marker: {
            color: models.map(m => m.f1_score * 100),
            colorscale: 'Viridis'
        },
        text: models.map(m => `${(m.f1_score * 100).toFixed(1)}%`),
        textposition: 'outside'
    };

    const layout = {
        title: 'F1 Score Comparison',
        xaxis: { title: 'Model' },
        yaxis: { title: 'F1 Score (%)', range: [85, 100] },
        height: 350
    };

    const config = { responsive: true, displayModeBar: true, displaylogo: false };

    Plotly.newPlot(containerId, [trace], layout, config);
}

/**
 * Create Comparison Radar Chart
 */
function createComparisonRadarChart(selectedModels, containerId) {
    const models = selectedModels.map(key => mockModelsData[key]).filter(m => m);
    const colors = [colorScheme.primary, colorScheme.success, colorScheme.warning, colorScheme.danger];

    const traces = models.map((model, idx) => ({
        type: 'scatterpolar',
        r: [
            model.f1_score * 100,
            model.precision * 100,
            model.recall * 100,
            model.roc_auc * 100,
            model.f1_score * 100
        ],
        theta: ['F1', 'Precision', 'Recall', 'ROC-AUC', 'F1'],
        fill: 'toself',
        name: model.name,
        line: { color: colors[idx % colors.length] },
        opacity: 0.6
    }));

    const layout = {
        title: 'Multi-Metric Radar Comparison',
        polar: {
            radialaxis: { visible: true, range: [85, 100] }
        },
        height: 350,
        showlegend: true
    };

    const config = { responsive: true, displayModeBar: true, displaylogo: false };

    Plotly.newPlot(containerId, traces, layout, config);
}

/**
 * Export Research Results
 */
function exportResearchResults() {
    const selectedModels = Array.from(document.querySelectorAll('.research-model-checkbox:checked'))
        .map(cb => cb.value);

    const exportData = {
        timestamp: new Date().toISOString(),
        analysis_type: 'research_mode_comparison',
        models_compared: selectedModels,
        results: selectedModels.map(key => {
            const model = mockModelsData[key];
            return {
                model: model.name,
                metrics: {
                    f1_score: model.f1_score,
                    precision: model.precision,
                    recall: model.recall,
                    roc_auc: model.roc_auc
                },
                performance: {
                    parameters: model.parameters,
                    inference_speed: model.inference_speed,
                    memory_usage: model.memory_usage,
                    training_time: model.training_time
                }
            };
        })
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `research_comparison_${Date.now()}.json`;
    link.click();

    showNotification('Research results exported!', 'success');
}

function loadComparisonExample() {
    showNotification('Loading example data...', 'info');
    setTimeout(() => {
        runResearchComparison();
    }, 500);
}

// Export all functions
window.runDetailedComparison = runDetailedComparison;
window.exportComparisonResults = exportComparisonResults;
window.exportAsJSON = exportAsJSON;
window.exportAsCSV = exportAsCSV;
window.exportAsPDF = exportAsPDF;
window.exportCharts = exportCharts;
window.enhanceResearchModeWithComparison = enhanceResearchModeWithComparison;
window.runResearchComparison = runResearchComparison;
window.exportResearchResults = exportResearchResults;
window.loadComparisonExample = loadComparisonExample;


/**
 * Educational Light Curve Analyzer Functions
 */
let currentEducationalSample = null;

function loadEducationalSample(type) {
    showNotification(`Loading ${type} sample...`, 'info');

    let hasPlanet;
    let difficulty;

    if (type === 'planet') {
        hasPlanet = true;
        difficulty = 'easy';
    } else if (type === 'no-planet') {
        hasPlanet = false;
        difficulty = 'easy';
    } else {
        hasPlanet = Math.random() > 0.5;
        difficulty = ['easy', 'medium', 'hard'][Math.floor(Math.random() * 3)];
    }

    // Generate light curve
    const lightCurve = generateLightCurve(hasPlanet, difficulty);
    currentEducationalSample = {
        ...lightCurve,
        hasPlanet,
        difficulty
    };

    // Plot the light curve
    plotEducationalLightCurve(lightCurve.time, lightCurve.flux);

    // Analyze and show results
    analyzeEducationalSample(lightCurve, hasPlanet);

    showNotification(`${type} sample loaded!`, 'success');
}

function plotEducationalLightCurve(time, flux) {
    const trace = {
        x: time,
        y: flux,
        mode: 'lines+markers',
        type: 'scatter',
        name: 'Light Curve',
        line: { color: '#3498db', width: 1 },
        marker: { size: 3, color: '#2980b9' }
    };

    const layout = {
        title: '📊 Star Brightness Over Time',
        xaxis: {
            title: 'Time (days)',
            gridcolor: '#ecf0f1'
        },
        yaxis: {
            title: 'Normalized Flux (Brightness)',
            gridcolor: '#ecf0f1'
        },
        height: 500,
        hovermode: 'closest',
        plot_bgcolor: '#f8f9fa',
        paper_bgcolor: 'white'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    Plotly.newPlot('educational-lightcurve-plot', [trace], layout, config);
}

function analyzeEducationalSample(lightCurve, hasPlanet) {
    const resultsPanel = document.getElementById('analysis-results-panel');
    if (!resultsPanel) return;

    // Calculate statistics
    const flux = lightCurve.flux;
    const mean = flux.reduce((a, b) => a + b, 0) / flux.length;
    const variance = flux.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / flux.length;
    const stdDev = Math.sqrt(variance);
    const minFlux = Math.min(...flux);
    const transitDepth = ((mean - minFlux) / mean) * 100;

    // Simulate AI confidence
    const confidence = hasPlanet ? (85 + Math.random() * 12) : (15 + Math.random() * 20);

    // Estimate period (simplified)
    const estimatedPeriod = hasPlanet ? (2 + Math.random() * 8).toFixed(2) : 'N/A';

    // Update results
    document.getElementById('detection-result').innerHTML = hasPlanet ?
        '<span class="planet-detected">🪐 Planet Detected!</span>' :
        '<span class="no-planet">⭐ No Planet</span>';

    document.getElementById('confidence-result').innerHTML =
        `<span class="confidence-value">${confidence.toFixed(1)}%</span>`;

    document.getElementById('depth-result').innerHTML =
        `<span class="depth-value">${transitDepth.toFixed(3)}%</span>`;

    document.getElementById('period-result').innerHTML =
        `<span class="period-value">${estimatedPeriod} days</span>`;

    resultsPanel.classList.remove('hidden');
}

function showInfoTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.info-tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.info-tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');
}

// Export functions
window.loadEducationalSample = loadEducationalSample;
window.showInfoTab = showInfoTab;
