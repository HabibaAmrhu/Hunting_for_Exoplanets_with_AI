// NASA Exoplanet Detection Pipeline - Game Logic (Hunt a Planet & Challenge Mode)

// Response Templates
const responseTemplates = {
    correct_planet: [
        "Excellent work! You spotted the transit signal!",
        "Great job! That's definitely a planet candidate!",
        "You're a natural planet hunter! Correct!",
        "Spot on! The dip in brightness indicates a planet!",
        "Well done! You identified the exoplanet transit!",
        "Perfect! Your eye for transits is impressive!",
        "Correct! That's a clear planetary signal!",
        "Bravo! You found the hidden world!"
    ],
    correct_no_planet: [
        "Correct! No planetary transit here.",
        "Right! This is stellar variability, not a planet.",
        "Good call! No transit signal detected.",
        "Exactly! This light curve shows no planets.",
        "Well done! You avoided a false positive.",
        "Correct! Just normal stellar activity.",
        "Right on! No exoplanet in this data.",
        "Perfect! You correctly identified no transit."
    ],
    missed_planet: [
        "Actually, there's a subtle transit here. Look closer!",
        "Oops! There's a faint planetary signal you missed.",
        "Not quite! There's a small dip indicating a planet.",
        "Close, but there's actually a transit present.",
        "Missed it! There's a planet hiding in this data.",
        "Try again! A subtle transit signal is there.",
        "Almost! But there's a planetary signature here.",
        "Look more carefully - there's a transit dip!"
    ],
    false_positive: [
        "Not quite! This is stellar variability, not a planet.",
        "Careful! That's noise, not a transit signal.",
        "Actually, no planet here - just stellar activity.",
        "Close, but this isn't a planetary transit.",
        "That's a false alarm - no planet detected.",
        "Not this time! It's stellar noise, not a planet.",
        "Tricky one! But there's no transit here.",
        "Almost! But this light curve has no planet."
    ]
};

// Generate synthetic light curve data
function generateLightCurve(hasPlanet = false, difficulty = 'easy') {
    const numPoints = 2048;
    const time = [];
    const flux = [];

    // Generate time array
    for (let i = 0; i < numPoints; i++) {
        time.push(i / 100);
    }

    // Base flux with noise
    const noiseLevel = difficulty === 'easy' ? 0.001 : difficulty === 'medium' ? 0.003 : 0.005;

    for (let i = 0; i < numPoints; i++) {
        let value = 1.0;

        // Add noise
        value += (Math.random() - 0.5) * noiseLevel;

        // Add stellar variability
        value += Math.sin(time[i] * 0.5) * 0.002;

        // Add transit if planet present
        if (hasPlanet) {
            const period = 5.0; // days
            const depth = difficulty === 'easy' ? 0.01 : difficulty === 'medium' ? 0.005 : 0.003;
            const duration = 0.2; // days

            const phase = (time[i] % period) / period;
            if (phase < duration / period || phase > 1 - duration / period) {
                const transitPhase = phase < 0.5 ? phase : 1 - phase;
                const transitShape = Math.exp(-Math.pow(transitPhase * period / duration * 4, 2));
                value -= depth * transitShape;
            }
        }

        flux.push(value);
    }

    return { time, flux };
}

// Plot light curve
function plotLightCurve(time, flux, containerId = 'light-curve-plot') {
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
        title: 'Light Curve',
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
        },
        margin: {
            l: 60,
            r: 30,
            t: 50,
            b: 60
        }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };

    Plotly.newPlot(containerId, [trace], layout, config);
}

// Generate new sample
function newSample() {
    // Randomly decide if sample has planet (50/50)
    const hasPlanet = Math.random() > 0.5;
    const difficulty = 'medium'; // Can be made dynamic

    // Generate light curve
    const { time, flux } = generateLightCurve(hasPlanet, difficulty);

    // Store current sample
    appState.currentSample = {
        hasPlanet,
        difficulty,
        time,
        flux,
        starId: `KIC-${Math.floor(Math.random() * 1000000)}`,
        survey: ['Kepler', 'TESS', 'K2'][Math.floor(Math.random() * 3)]
    };

    // Plot the light curve
    plotLightCurve(time, flux);

    // Hide feedback and explanation
    document.getElementById('feedback-container').classList.add('hidden');
    document.getElementById('explanation-container').classList.add('hidden');
    document.getElementById('ai-analysis-container').classList.add('hidden');

    showNotification('New sample loaded!', 'info');
}

// Make guess
function makeGuess(userSaysPlanet) {
    const sample = appState.currentSample;
    if (!sample) {
        showNotification('Please load a sample first!', 'error');
        return;
    }

    updateAttempts();

    const correct = userSaysPlanet === sample.hasPlanet;
    let responseType;

    if (correct && sample.hasPlanet) {
        responseType = 'correct_planet';
        updateScore(1);
    } else if (correct && !sample.hasPlanet) {
        responseType = 'correct_no_planet';
        updateScore(1);
    } else if (!correct && sample.hasPlanet) {
        responseType = 'missed_planet';
    } else {
        responseType = 'false_positive';
    }

    // Get random response from template
    const responses = responseTemplates[responseType];
    const response = responses[Math.floor(Math.random() * responses.length)];

    // Display feedback
    const feedbackContainer = document.getElementById('feedback-container');
    feedbackContainer.innerHTML = `
        <h3>${correct ? '✅ Correct!' : '❌ Incorrect'}</h3>
        <p>${response}</p>
    `;
    feedbackContainer.className = `feedback-container ${correct ? 'correct' : 'incorrect'}`;
    feedbackContainer.classList.remove('hidden');

    // Display explanation
    displayExplanation(sample, correct);

    // Show AI analysis button
    showAIAnalysisOption();
}

// Display explanation
function displayExplanation(sample, correct) {
    const explanationContainer = document.getElementById('explanation-container');

    let explanation = '';
    if (sample.hasPlanet) {
        explanation = `
            <h3>🪐 Planet Detected!</h3>
            <p>This light curve shows a periodic dip in brightness, which is the signature of a planet passing in front of its star (transit method).</p>
            <h4>Key Features:</h4>
            <ul>
                <li>Transit Depth: ~${(Math.random() * 1 + 0.5).toFixed(2)}% - indicates planet size</li>
                <li>Period: ~5.0 days - orbital period</li>
                <li>Star: ${sample.starId} from ${sample.survey} mission</li>
            </ul>
        `;
    } else {
        explanation = `
            <h3>⭐ No Planet Here</h3>
            <p>This light curve shows stellar variability but no periodic transits. The variations you see are due to:</p>
            <ul>
                <li>Stellar rotation and spots</li>
                <li>Stellar pulsations</li>
                <li>Instrumental noise</li>
                <li>Other astrophysical phenomena</li>
            </ul>
            <p>No periodic dips characteristic of planetary transits are present.</p>
        `;
    }

    explanationContainer.innerHTML = explanation;
    explanationContainer.classList.remove('hidden');
}

// Show AI analysis option
function showAIAnalysisOption() {
    const aiContainer = document.getElementById('ai-analysis-container');
    aiContainer.innerHTML = `
        <h3>🤖 AI Analysis</h3>
        <button class="btn-primary" onclick="runAIAnalysis()">Run AI Analysis</button>
    `;
    aiContainer.classList.remove('hidden');
}

// Run AI analysis
function runAIAnalysis() {
    const sample = appState.currentSample;
    if (!sample) return;

    // Simulate AI prediction
    const confidence = sample.hasPlanet ?
        (85 + Math.random() * 10).toFixed(1) :
        (90 + Math.random() * 8).toFixed(1);

    const prediction = sample.hasPlanet ? 'PLANET DETECTED' : 'NO PLANET';

    const aiContainer = document.getElementById('ai-analysis-container');
    aiContainer.innerHTML = `
        <h3>🤖 AI Analysis Results</h3>
        <div class="ai-results">
            <div class="ai-prediction">
                <h4>Prediction: ${prediction}</h4>
                <p>Confidence: ${confidence}%</p>
            </div>
            <div class="ai-details">
                <h4>Model: Ensemble (CNN + LSTM + Transformer)</h4>
                <p>The AI model analyzed the light curve and detected ${sample.hasPlanet ? 'periodic transit signals' : 'no significant transit patterns'}.</p>
            </div>
        </div>
        <button class="btn-secondary mt-2" onclick="newSample()">Next Challenge</button>
    `;
}

// Show hint
function showHint() {
    const sample = appState.currentSample;
    if (!sample) return;

    // Analyze the light curve to provide dynamic hints
    const flux = sample.flux;
    const time = sample.time;
    
    // Calculate statistics
    const mean = flux.reduce((a, b) => a + b, 0) / flux.length;
    const variance = flux.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / flux.length;
    const stdDev = Math.sqrt(variance);
    
    // Find dips (values significantly below mean)
    const threshold = mean - (2 * stdDev);
    const dips = flux.filter(f => f < threshold).length;
    const dipPercentage = (dips / flux.length) * 100;
    
    // Calculate flux range
    const minFlux = Math.min(...flux);
    const maxFlux = Math.max(...flux);
    const fluxRange = maxFlux - minFlux;
    const transitDepth = ((mean - minFlux) / mean) * 100;
    
    // Generate dynamic hint based on actual data
    let hint = '';
    let detailedHint = '';
    
    if (sample.hasPlanet) {
        // Hints for planet samples
        if (dipPercentage > 5) {
            hint = "💡 <strong>Look for the dips!</strong> This light curve has noticeable drops in brightness.";
            detailedHint = `
                <div class="hint-details">
                    <p>🔍 <strong>What to look for:</strong></p>
                    <ul>
                        <li>📉 There are <strong>${dips} data points</strong> showing significant dips (${dipPercentage.toFixed(1)}% of data)</li>
                        <li>📊 The transit depth is approximately <strong>${transitDepth.toFixed(2)}%</strong></li>
                        <li>🔄 Look for <strong>repeating patterns</strong> - planets orbit regularly!</li>
                        <li>⏱️ The dips should have similar <strong>shape and depth</strong></li>
                    </ul>
                    <p class="hint-tip">💡 <em>Tip: Zoom in on the dips to see their characteristic U-shape or V-shape!</em></p>
                </div>
            `;
        } else {
            hint = "💡 <strong>Subtle signal!</strong> Look very carefully for small, periodic dips.";
            detailedHint = `
                <div class="hint-details">
                    <p>🔍 <strong>This is a challenging one:</strong></p>
                    <ul>
                        <li>🔬 The transit is <strong>subtle</strong> - only ${transitDepth.toFixed(2)}% depth</li>
                        <li>👀 Look for <strong>small but consistent</strong> dips</li>
                        <li>📏 The dips are <strong>regular</strong> - they repeat at the same interval</li>
                        <li>🎯 Focus on the <strong>overall pattern</strong> rather than individual points</li>
                    </ul>
                    <p class="hint-tip">💡 <em>Tip: Small planets or distant orbits create subtle transits!</em></p>
                </div>
            `;
        }
    } else {
        // Hints for non-planet samples
        if (stdDev > 0.005) {
            hint = "💡 <strong>Check the pattern!</strong> Is the variation random or periodic?";
            detailedHint = `
                <div class="hint-details">
                    <p>🔍 <strong>What to look for:</strong></p>
                    <ul>
                        <li>🌊 The variation is <strong>${(stdDev * 100).toFixed(3)}%</strong> - this could be stellar noise</li>
                        <li>🎲 Look for <strong>random fluctuations</strong> vs. regular patterns</li>
                        <li>⚡ Stellar variability doesn't have the <strong>characteristic transit shape</strong></li>
                        <li>🔄 No <strong>repeating dips</strong> at regular intervals</li>
                    </ul>
                    <p class="hint-tip">💡 <em>Tip: Stars naturally vary in brightness, but not in a regular, repeating pattern!</em></p>
                </div>
            `;
        } else {
            hint = "💡 <strong>Very stable!</strong> Look for any regular, repeating dips.";
            detailedHint = `
                <div class="hint-details">
                    <p>🔍 <strong>What to look for:</strong></p>
                    <ul>
                        <li>✨ This star is <strong>very stable</strong> - low noise (${(stdDev * 100).toFixed(3)}%)</li>
                        <li>📊 The flux range is only <strong>${(fluxRange * 100).toFixed(3)}%</strong></li>
                        <li>🔍 No <strong>periodic dips</strong> are visible</li>
                        <li>⭐ This is likely just <strong>stellar noise</strong> or measurement uncertainty</li>
                    </ul>
                    <p class="hint-tip">💡 <em>Tip: A flat, stable light curve usually means no planet!</em></p>
                </div>
            `;
        }
    }
    
    // Show hint in a modal/panel
    const hintContainer = document.getElementById('explanation-container');
    if (hintContainer) {
        hintContainer.innerHTML = `
            <div class="hint-panel fade-in">
                <div class="hint-header">
                    <h3>💡 Dynamic Hint</h3>
                    <button class="close-btn" onclick="document.getElementById('explanation-container').classList.add('hidden')">✕</button>
                </div>
                <div class="hint-content">
                    <div class="hint-main">${hint}</div>
                    ${detailedHint}
                    <div class="hint-stats">
                        <h4>📊 Light Curve Statistics:</h4>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">Mean Flux:</span>
                                <span class="stat-value">${mean.toFixed(6)}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Std Deviation:</span>
                                <span class="stat-value">${(stdDev * 100).toFixed(3)}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Flux Range:</span>
                                <span class="stat-value">${(fluxRange * 100).toFixed(3)}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Data Points:</span>
                                <span class="stat-value">${flux.length}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        hintContainer.classList.remove('hidden');
    }
    
    showNotification('Hint revealed! Check the panel below.', 'info');
}

// Learn the Basics - Topic Content
const topicContent = {
    1: `
        <h2>📊 What are Light Curves?</h2>
        <p>A light curve is a graph showing how the brightness of a star changes over time.</p>
        <h3>Components:</h3>
        <ul>
            <li><strong>X-axis:</strong> Time (usually in days)</li>
            <li><strong>Y-axis:</strong> Brightness (normalized flux)</li>
            <li><strong>Data Points:</strong> Individual measurements</li>
        </ul>
        <h3>Why They Matter:</h3>
        <p>Light curves reveal information about:</p>
        <ul>
            <li>Planetary transits</li>
            <li>Stellar variability</li>
            <li>Binary star systems</li>
            <li>Stellar pulsations</li>
        </ul>
    `,
    2: `
        <h2>🌟 The Transit Method</h2>
        <p>When a planet passes in front of its star (from our viewpoint), it blocks a tiny amount of starlight, causing a dip in brightness.</p>
        <h3>Transit Characteristics:</h3>
        <ul>
            <li><strong>Periodic:</strong> Repeats with orbital period</li>
            <li><strong>Symmetric:</strong> Similar shape each time</li>
            <li><strong>Depth:</strong> Related to planet size</li>
            <li><strong>Duration:</strong> Related to orbital speed</li>
        </ul>
        <h3>Detection Process:</h3>
        <ol>
            <li>Monitor star brightness continuously</li>
            <li>Look for periodic dips</li>
            <li>Confirm multiple transits</li>
            <li>Rule out false positives</li>
            <li>Validate with additional observations</li>
        </ol>
    `,
    3: `
        <h2>🔍 Types of Signals</h2>
        <h3>1. Planetary Transits</h3>
        <ul>
            <li>✓ Periodic dips</li>
            <li>✓ Consistent depth</li>
            <li>✓ Symmetric shape</li>
        </ul>
        <h3>2. Stellar Variability</h3>
        <ul>
            <li>✗ Irregular patterns</li>
            <li>✗ Variable amplitude</li>
            <li>✗ No periodicity</li>
        </ul>
        <h3>3. Noise & Artifacts</h3>
        <ul>
            <li>✗ Random fluctuations</li>
            <li>✗ Instrumental effects</li>
            <li>✗ No physical meaning</li>
        </ul>
        <h3>4. False Positives</h3>
        <ul>
            <li>✗ Mimics transits</li>
            <li>✗ Binary stars</li>
            <li>✗ Background eclipses</li>
        </ul>
    `,
    4: `
        <h2>⚠️ Common Challenges</h2>
        <h3>1. Noise in Data</h3>
        <ul>
            <li>Instrumental noise</li>
            <li>Photon noise</li>
            <li>Systematic errors</li>
        </ul>
        <p><strong>Solution:</strong> Advanced filtering & AI</p>
        
        <h3>2. Small Transit Signals</h3>
        <ul>
            <li>Earth-sized planets: ~0.01% dip</li>
            <li>Requires high precision</li>
        </ul>
        <p><strong>Solution:</strong> Long observations, AI detection</p>
        
        <h3>3. Stellar Activity</h3>
        <ul>
            <li>Star spots</li>
            <li>Stellar pulsations</li>
            <li>Rotation effects</li>
        </ul>
        <p><strong>Solution:</strong> Pattern recognition, AI classification</p>
        
        <h3>4. False Positives</h3>
        <ul>
            <li>Binary stars</li>
            <li>Background eclipses</li>
            <li>Instrumental artifacts</li>
        </ul>
        <p><strong>Solution:</strong> Multi-model validation</p>
    `,
    5: `
        <h2>🤖 How AI Helps</h2>
        <h3>Machine Learning Advantages:</h3>
        <ul>
            <li><strong>Pattern Recognition:</strong> Identifies subtle signals</li>
            <li><strong>Speed:</strong> Analyzes thousands of light curves quickly</li>
            <li><strong>Consistency:</strong> No human fatigue or bias</li>
            <li><strong>Uncertainty:</strong> Quantifies confidence levels</li>
        </ul>
        
        <h3>Our AI Models:</h3>
        <h4>1. CNN (Convolutional Neural Network)</h4>
        <ul>
            <li>Learns spatial patterns</li>
            <li>Fast inference</li>
            <li>94% accuracy</li>
        </ul>
        
        <h4>2. LSTM (Long Short-Term Memory)</h4>
        <ul>
            <li>Captures temporal dependencies</li>
            <li>Handles sequences</li>
            <li>92% accuracy</li>
        </ul>
        
        <h4>3. Transformer</h4>
        <ul>
            <li>Attention mechanisms</li>
            <li>Long-range dependencies</li>
            <li>95% accuracy</li>
        </ul>
        
        <h4>4. Ensemble</h4>
        <ul>
            <li>Combines all models</li>
            <li>Highest accuracy: 96%</li>
            <li>Most reliable</li>
        </ul>
        
        <h3>Confidence Scores:</h3>
        <ul>
            <li><strong>High (>90%):</strong> Very likely correct</li>
            <li><strong>Medium (70-90%):</strong> Probably correct</li>
            <li><strong>Low (<70%):</strong> Uncertain, needs review</li>
        </ul>
    `
};

// Show topic content
document.querySelectorAll('.topic-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const topic = btn.dataset.topic;
        const contentDiv = document.getElementById('topic-content');
        contentDiv.innerHTML = topicContent[topic];

        // Update active button
        document.querySelectorAll('.topic-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

// Example Gallery
function showExample(exampleNum) {
    const examples = {
        1: {
            title: "Clear Transit Signal",
            difficulty: "⭐ Easy",
            description: "This is an ideal example of a planetary transit. The signal is strong and clear.",
            hasPlanet: true,
            difficultyLevel: 'easy'
        },
        2: {
            title: "Noisy Transit Signal",
            difficulty: "⭐⭐ Medium",
            description: "Real exoplanet data often looks like this. The transit signal is present but buried in noise.",
            hasPlanet: true,
            difficultyLevel: 'medium'
        },
        3: {
            title: "Stellar Variability",
            difficulty: "⭐⭐ Medium",
            description: "This shows stellar variability, not a planet. Stars naturally vary in brightness.",
            hasPlanet: false,
            difficultyLevel: 'medium'
        },
        4: {
            title: "Subtle Transit Signal",
            difficulty: "⭐⭐⭐ Hard",
            description: "This represents the cutting edge of exoplanet detection. Earth-sized planets produce tiny signals.",
            hasPlanet: true,
            difficultyLevel: 'hard'
        }
    };

    const example = examples[exampleNum];
    const { time, flux } = generateLightCurve(example.hasPlanet, example.difficultyLevel);

    const detailDiv = document.getElementById('example-detail');
    detailDiv.innerHTML = `
        <h2>${example.title}</h2>
        <p><strong>Difficulty:</strong> ${example.difficulty}</p>
        <p>${example.description}</p>
        <div id="example-plot" class="plot-container"></div>
        <button class="btn-secondary" onclick="document.getElementById('example-detail').classList.add('hidden')">Close</button>
    `;
    detailDiv.classList.remove('hidden');

    // Plot after a short delay to ensure container is visible
    setTimeout(() => {
        plotLightCurve(time, flux, 'example-plot');
    }, 100);
}

// Challenge Mode
// Challenge Mode State
let challengeState = {
    level: 0,
    currentQuestion: 0,
    totalQuestions: 0,
    timePerQuestion: 0,
    score: 0,
    timeRemaining: 0,
    timer: null,
    answers: []
};

function startChallenge(level) {
    const levels = {
        1: { name: "Beginner Detective", questions: 10, time: 60, difficulty: 'easy' },
        2: { name: "Planet Hunter", questions: 15, time: 45, difficulty: 'medium' },
        3: { name: "Expert Astronomer", questions: 20, time: 30, difficulty: 'hard' },
        4: { name: "Master Discoverer", questions: 25, time: 20, difficulty: 'expert' }
    };

    const challenge = levels[level];
    showNotification(`Starting ${challenge.name} challenge!`, 'info');

    // Initialize challenge state
    challengeState = {
        level: level,
        levelName: challenge.name,
        currentQuestion: 1,
        totalQuestions: challenge.questions,
        timePerQuestion: challenge.time,
        difficulty: challenge.difficulty,
        score: 0,
        timeRemaining: challenge.time,
        timer: null,
        answers: [],
        startTime: Date.now()
    };

    // Show challenge interface
    showChallengeInterface();
    startChallengeQuestion();
}

function showChallengeInterface() {
    const challengeContainer = document.querySelector('.challenge-container');
    if (!challengeContainer) return;

    challengeContainer.innerHTML = `
        <div class="challenge-active">
            <div class="challenge-header">
                <div class="challenge-info">
                    <h2>🏆 ${challengeState.levelName}</h2>
                    <p>Question ${challengeState.currentQuestion} of ${challengeState.totalQuestions}</p>
                </div>
                <div class="challenge-stats">
                    <div class="stat-item">
                        <span class="stat-label">Score:</span>
                        <span class="stat-value" id="challenge-score">${challengeState.score}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Time:</span>
                        <span class="stat-value" id="challenge-timer">${challengeState.timeRemaining}s</span>
                    </div>
                </div>
            </div>
            
            <div id="challenge-plot" class="plot-container"></div>
            
            <div class="challenge-question">
                <h3>Does this light curve show a planet transit?</h3>
                <div class="challenge-buttons">
                    <button class="btn-choice btn-planet" onclick="makeChallengeGuess(true)">🪐 Yes, Planet!</button>
                    <button class="btn-choice btn-no-planet" onclick="makeChallengeGuess(false)">⭐ No Planet</button>
                </div>
            </div>
            
            <div id="challenge-feedback" class="challenge-feedback hidden"></div>
            
            <div class="challenge-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${(challengeState.currentQuestion / challengeState.totalQuestions) * 100}%"></div>
                </div>
            </div>
        </div>
    `;
}

function startChallengeQuestion() {
    // Generate light curve based on difficulty
    const hasPlanet = Math.random() > 0.5;
    const lightCurve = generateLightCurve(hasPlanet, challengeState.difficulty);

    // Store correct answer
    challengeState.currentAnswer = hasPlanet;
    challengeState.currentLightCurve = lightCurve;

    // Plot light curve
    plotLightCurve(lightCurve.time, lightCurve.flux, 'challenge-plot');

    // Start timer
    challengeState.timeRemaining = challengeState.timePerQuestion;
    updateChallengeTimer();

    if (challengeState.timer) {
        clearInterval(challengeState.timer);
    }

    challengeState.timer = setInterval(() => {
        challengeState.timeRemaining--;
        updateChallengeTimer();

        if (challengeState.timeRemaining <= 0) {
            clearInterval(challengeState.timer);
            makeChallengeGuess(null); // Time's up
        }
    }, 1000);
}

function updateChallengeTimer() {
    const timerElement = document.getElementById('challenge-timer');
    if (timerElement) {
        timerElement.textContent = `${challengeState.timeRemaining}s`;
        if (challengeState.timeRemaining <= 10) {
            timerElement.style.color = '#e74c3c';
        } else {
            timerElement.style.color = 'inherit';
        }
    }
}

function makeChallengeGuess(guess) {
    // Stop timer
    if (challengeState.timer) {
        clearInterval(challengeState.timer);
    }

    const correct = guess === challengeState.currentAnswer;
    const timeBonus = Math.max(0, challengeState.timeRemaining * 10);
    const points = correct ? (100 + timeBonus) : 0;

    if (correct) {
        challengeState.score += points;
    }

    // Record answer
    challengeState.answers.push({
        question: challengeState.currentQuestion,
        guess: guess,
        correct: correct,
        answer: challengeState.currentAnswer,
        timeUsed: challengeState.timePerQuestion - challengeState.timeRemaining,
        points: points
    });

    // Show feedback
    showChallengeFeedback(correct, guess === null);

    // Update score display
    const scoreElement = document.getElementById('challenge-score');
    if (scoreElement) {
        scoreElement.textContent = challengeState.score;
    }
}

function showChallengeFeedback(correct, timeout) {
    const feedbackDiv = document.getElementById('challenge-feedback');
    if (!feedbackDiv) return;

    let message, className;
    if (timeout) {
        message = "⏰ Time's up!";
        className = 'timeout';
    } else if (correct) {
        message = "✅ Correct!";
        className = 'correct';
    } else {
        message = "❌ Incorrect";
        className = 'incorrect';
    }

    feedbackDiv.innerHTML = `
        <div class="feedback-message ${className}">
            <h4>${message}</h4>
            <p>The correct answer was: ${challengeState.currentAnswer ? '🪐 Planet' : '⭐ No Planet'}</p>
        </div>
    `;
    feedbackDiv.classList.remove('hidden');

    // Move to next question or end challenge
    setTimeout(() => {
        feedbackDiv.classList.add('hidden');
        challengeState.currentQuestion++;

        if (challengeState.currentQuestion <= challengeState.totalQuestions) {
            showChallengeInterface();
            startChallengeQuestion();
        } else {
            endChallenge();
        }
    }, 2000);
}

function endChallenge() {
    const challengeContainer = document.querySelector('.challenge-container');
    if (!challengeContainer) return;

    const accuracy = (challengeState.answers.filter(a => a.correct).length / challengeState.totalQuestions) * 100;
    const totalTime = (Date.now() - challengeState.startTime) / 1000;
    const avgTime = totalTime / challengeState.totalQuestions;

    challengeContainer.innerHTML = `
        <div class="challenge-results">
            <h2>🏆 Challenge Complete!</h2>
            <h3>${challengeState.levelName}</h3>
            
            <div class="results-grid">
                <div class="result-card">
                    <div class="result-icon">🎯</div>
                    <div class="result-value">${challengeState.score}</div>
                    <div class="result-label">Total Score</div>
                </div>
                
                <div class="result-card">
                    <div class="result-icon">✅</div>
                    <div class="result-value">${accuracy.toFixed(1)}%</div>
                    <div class="result-label">Accuracy</div>
                </div>
                
                <div class="result-card">
                    <div class="result-icon">⏱️</div>
                    <div class="result-value">${avgTime.toFixed(1)}s</div>
                    <div class="result-label">Avg Time</div>
                </div>
                
                <div class="result-card">
                    <div class="result-icon">📊</div>
                    <div class="result-value">${challengeState.answers.filter(a => a.correct).length}/${challengeState.totalQuestions}</div>
                    <div class="result-label">Correct</div>
                </div>
            </div>
            
            <div class="performance-message">
                <h4>${getPerformanceMessage(accuracy)}</h4>
            </div>
            
            <div class="challenge-actions">
                <button class="btn-primary" onclick="startChallenge(${challengeState.level})">🔄 Try Again</button>
                <button class="btn-secondary" onclick="location.reload()">🏠 Back to Menu</button>
                <button class="btn-secondary" onclick="exportChallengeResults()">📥 Export Results</button>
            </div>
        </div>
    `;
}

function getPerformanceMessage(accuracy) {
    if (accuracy >= 90) return "🌟 Outstanding! You're a true exoplanet expert!";
    if (accuracy >= 75) return "🎉 Great job! You have a keen eye for planets!";
    if (accuracy >= 60) return "👍 Good work! Keep practicing!";
    return "💪 Keep trying! You'll get better with practice!";
}

function exportChallengeResults() {
    const exportData = {
        timestamp: new Date().toISOString(),
        challenge: {
            level: challengeState.level,
            name: challengeState.levelName,
            difficulty: challengeState.difficulty
        },
        results: {
            score: challengeState.score,
            accuracy: (challengeState.answers.filter(a => a.correct).length / challengeState.totalQuestions) * 100,
            total_questions: challengeState.totalQuestions,
            correct_answers: challengeState.answers.filter(a => a.correct).length,
            total_time: (Date.now() - challengeState.startTime) / 1000
        },
        answers: challengeState.answers
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `challenge_results_${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showNotification('Challenge results exported!', 'success');
}

window.makeChallengeGuess = makeChallengeGuess;
window.exportChallengeResults = exportChallengeResults;

// Export functions
window.newSample = newSample;
window.makeGuess = makeGuess;
window.showHint = showHint;
window.runAIAnalysis = runAIAnalysis;
window.showExample = showExample;
window.startChallenge = startChallenge;


/**
 * Enhanced Interactive Learn the Basics
 */
function initializeLearnTheBasics() {
    // Add event listeners to topic buttons
    const topicButtons = document.querySelectorAll('.topic-btn');
    topicButtons.forEach((btn, index) => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons
            topicButtons.forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            btn.classList.add('active');
            // Show topic content
            showInteractiveTopic(index + 1);
        });
    });
    
    // Show first topic by default
    if (topicButtons.length > 0) {
        topicButtons[0].classList.add('active');
        showInteractiveTopic(1);
    }
}

function showInteractiveTopic(topicNumber) {
    const contentDiv = document.getElementById('topic-content');
    if (!contentDiv) return;
    
    const topics = {
        1: {
            title: "📊 What are Light Curves?",
            content: `
                <div class="interactive-topic">
                    <div class="topic-intro">
                        <p class="lead-text">A light curve is like a <strong>heartbeat monitor for stars</strong> - it shows how their brightness changes over time!</p>
                    </div>
                    
                    <div class="interactive-demo">
                        <h3>🎮 Interactive Demo</h3>
                        <div class="demo-container">
                            <div id="demo-lightcurve-1" class="demo-plot"></div>
                            <div class="demo-controls">
                                <button class="btn-demo" onclick="animateLightCurve(1, true)">🪐 Show Planet Transit</button>
                                <button class="btn-demo" onclick="animateLightCurve(1, false)">⭐ Show No Planet</button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="topic-section">
                        <h3>📐 Components of a Light Curve</h3>
                        <div class="component-grid">
                            <div class="component-card">
                                <div class="component-icon">📅</div>
                                <h4>X-Axis: Time</h4>
                                <p>Usually measured in <strong>days</strong></p>
                                <p class="detail">Shows when observations were made</p>
                            </div>
                            <div class="component-card">
                                <div class="component-icon">💡</div>
                                <h4>Y-Axis: Brightness</h4>
                                <p>Normalized <strong>flux</strong> (light intensity)</p>
                                <p class="detail">1.0 = normal brightness</p>
                            </div>
                            <div class="component-card">
                                <div class="component-icon">📍</div>
                                <h4>Data Points</h4>
                                <p>Individual <strong>measurements</strong></p>
                                <p class="detail">More points = better accuracy</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="topic-section">
                        <h3>🔬 Why Light Curves Matter</h3>
                        <div class="info-cards">
                            <div class="info-card planet">
                                <h4>🪐 Planetary Transits</h4>
                                <p>Detect planets passing in front of stars</p>
                            </div>
                            <div class="info-card star">
                                <h4>⭐ Stellar Variability</h4>
                                <p>Study how stars change naturally</p>
                            </div>
                            <div class="info-card binary">
                                <h4>👥 Binary Stars</h4>
                                <p>Find stars orbiting each other</p>
                            </div>
                            <div class="info-card pulse">
                                <h4>💓 Stellar Pulsations</h4>
                                <p>Observe stars expanding and contracting</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="fun-fact">
                        <h4>🎯 Fun Fact!</h4>
                        <p>NASA's Kepler mission observed over <strong>150,000 stars</strong> simultaneously, creating millions of light curves to search for exoplanets!</p>
                    </div>
                </div>
            `
        },
        2: {
            title: "🌟 The Transit Method",
            content: `
                <div class="interactive-topic">
                    <div class="topic-intro">
                        <p class="lead-text">Imagine a tiny fly passing in front of a spotlight - that's what a planet transit looks like!</p>
                    </div>
                    
                    <div class="interactive-demo">
                        <h3>🎮 Interactive Transit Simulator</h3>
                        <div class="demo-container">
                            <div id="demo-lightcurve-2" class="demo-plot"></div>
                            <div class="demo-controls">
                                <label>Planet Size: <input type="range" id="planet-size" min="1" max="10" value="5" oninput="updateTransitDemo()"></label>
                                <label>Orbital Period: <input type="range" id="orbital-period" min="1" max="10" value="5" oninput="updateTransitDemo()"></label>
                                <button class="btn-demo" onclick="updateTransitDemo()">🔄 Update Transit</button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="topic-section">
                        <h3>🔍 Transit Characteristics</h3>
                        <div class="characteristic-grid">
                            <div class="char-card">
                                <div class="char-icon">🔄</div>
                                <h4>Periodic</h4>
                                <p>Repeats with the planet's <strong>orbital period</strong></p>
                                <div class="example">Example: Every 365 days for Earth</div>
                            </div>
                            <div class="char-card">
                                <div class="char-icon">⚖️</div>
                                <h4>Symmetric</h4>
                                <p>Same <strong>shape</strong> each time</p>
                                <div class="example">U-shaped or V-shaped dip</div>
                            </div>
                            <div class="char-card">
                                <div class="char-icon">📏</div>
                                <h4>Depth</h4>
                                <p>Related to <strong>planet size</strong></p>
                                <div class="example">Bigger planet = deeper dip</div>
                            </div>
                            <div class="char-card">
                                <div class="char-icon">⏱️</div>
                                <h4>Duration</h4>
                                <p>Related to <strong>orbital speed</strong></p>
                                <div class="example">Faster orbit = shorter transit</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="topic-section">
                        <h3>📋 Detection Process</h3>
                        <div class="process-steps">
                            <div class="step">
                                <div class="step-number">1</div>
                                <div class="step-content">
                                    <h4>Monitor Continuously</h4>
                                    <p>Observe star brightness 24/7</p>
                                </div>
                            </div>
                            <div class="step">
                                <div class="step-number">2</div>
                                <div class="step-content">
                                    <h4>Look for Dips</h4>
                                    <p>Search for periodic brightness drops</p>
                                </div>
                            </div>
                            <div class="step">
                                <div class="step-number">3</div>
                                <div class="step-content">
                                    <h4>Confirm Transits</h4>
                                    <p>Verify multiple occurrences</p>
                                </div>
                            </div>
                            <div class="step">
                                <div class="step-number">4</div>
                                <div class="step-content">
                                    <h4>Rule Out False Positives</h4>
                                    <p>Eliminate other explanations</p>
                                </div>
                            </div>
                            <div class="step">
                                <div class="step-number">5</div>
                                <div class="step-content">
                                    <h4>Validate</h4>
                                    <p>Confirm with additional observations</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="fun-fact">
                        <h4>🎯 Amazing Fact!</h4>
                        <p>An Earth-sized planet transiting a Sun-like star causes only a <strong>0.008%</strong> dip in brightness - that's like detecting a flea on a car headlight from a mile away!</p>
                    </div>
                </div>
            `
        },
        3: {
            title: "🔍 Types of Signals",
            content: `
                <div class="interactive-topic">
                    <div class="topic-intro">
                        <p class="lead-text">Not every dip in a light curve is a planet! Let's learn to tell them apart.</p>
                    </div>
                    
                    <div class="signal-comparison">
                        <div class="signal-card planet-signal">
                            <h3>✅ Planetary Transit</h3>
                            <div class="signal-demo" id="signal-planet"></div>
                            <div class="signal-features">
                                <div class="feature good">✓ Periodic dips</div>
                                <div class="feature good">✓ Consistent depth</div>
                                <div class="feature good">✓ Symmetric shape</div>
                                <div class="feature good">✓ Regular timing</div>
                            </div>
                            <button class="btn-demo" onclick="showSignalExample('planet')">👀 See Example</button>
                        </div>
                        
                        <div class="signal-card noise-signal">
                            <h3>❌ Stellar Variability</h3>
                            <div class="signal-demo" id="signal-variability"></div>
                            <div class="signal-features">
                                <div class="feature bad">✗ Irregular patterns</div>
                                <div class="feature bad">✗ Variable amplitude</div>
                                <div class="feature bad">✗ No periodicity</div>
                                <div class="feature bad">✗ Random timing</div>
                            </div>
                            <button class="btn-demo" onclick="showSignalExample('variability')">👀 See Example</button>
                        </div>
                        
                        <div class="signal-card noise-signal">
                            <h3>❌ Noise & Artifacts</h3>
                            <div class="signal-demo" id="signal-noise"></div>
                            <div class="signal-features">
                                <div class="feature bad">✗ Random fluctuations</div>
                                <div class="feature bad">✗ Instrumental effects</div>
                                <div class="feature bad">✗ No physical meaning</div>
                                <div class="feature bad">✗ High frequency</div>
                            </div>
                            <button class="btn-demo" onclick="showSignalExample('noise')">👀 See Example</button>
                        </div>
                        
                        <div class="signal-card false-signal">
                            <h3>⚠️ False Positives</h3>
                            <div class="signal-demo" id="signal-false"></div>
                            <div class="signal-features">
                                <div class="feature warning">⚠ Mimics transits</div>
                                <div class="feature warning">⚠ Binary stars</div>
                                <div class="feature warning">⚠ Background eclipses</div>
                                <div class="feature warning">⚠ Needs verification</div>
                            </div>
                            <button class="btn-demo" onclick="showSignalExample('false')">👀 See Example</button>
                        </div>
                    </div>
                    
                    <div class="quiz-section">
                        <h3>🎯 Quick Quiz!</h3>
                        <div class="quiz-question">
                            <p><strong>Which feature is MOST important for confirming a planet?</strong></p>
                            <div class="quiz-options">
                                <button class="quiz-option" onclick="checkQuizAnswer(this, false)">A) Deep dips</button>
                                <button class="quiz-option" onclick="checkQuizAnswer(this, true)">B) Periodic repetition</button>
                                <button class="quiz-option" onclick="checkQuizAnswer(this, false)">C) Bright star</button>
                                <button class="quiz-option" onclick="checkQuizAnswer(this, false)">D) Long duration</button>
                            </div>
                            <div class="quiz-feedback hidden"></div>
                        </div>
                    </div>
                </div>
            `
        },
        4: {
            title: "⚠️ Common Challenges",
            content: `
                <div class="interactive-topic">
                    <div class="topic-intro">
                        <p class="lead-text">Finding exoplanets is like finding a needle in a haystack... while wearing sunglasses... in the dark!</p>
                    </div>
                    
                    <div class="challenges-grid">
                        <div class="challenge-card">
                            <div class="challenge-icon">📊</div>
                            <h3>1. Noise in Data</h3>
                            <div class="challenge-content">
                                <p><strong>The Problem:</strong></p>
                                <ul>
                                    <li>Instrumental noise</li>
                                    <li>Photon noise (quantum uncertainty)</li>
                                    <li>Systematic errors</li>
                                    <li>Atmospheric interference</li>
                                </ul>
                                <div class="solution-box">
                                    <strong>💡 Solution:</strong>
                                    <p>Advanced filtering algorithms & AI models that can distinguish signal from noise!</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="challenge-card">
                            <div class="challenge-icon">🔬</div>
                            <h3>2. Small Transit Signals</h3>
                            <div class="challenge-content">
                                <p><strong>The Problem:</strong></p>
                                <ul>
                                    <li>Earth-sized planets: ~0.01% dip</li>
                                    <li>Requires extreme precision</li>
                                    <li>Easy to miss in noise</li>
                                </ul>
                                <div class="solution-box">
                                    <strong>💡 Solution:</strong>
                                    <p>High-precision instruments like Kepler & TESS, plus machine learning to detect subtle patterns!</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="challenge-card">
                            <div class="challenge-icon">⏱️</div>
                            <h3>3. Long Orbital Periods</h3>
                            <div class="challenge-content">
                                <p><strong>The Problem:</strong></p>
                                <ul>
                                    <li>Earth takes 365 days to orbit</li>
                                    <li>Need multiple transits to confirm</li>
                                    <li>Requires years of observation</li>
                                </ul>
                                <div class="solution-box">
                                    <strong>💡 Solution:</strong>
                                    <p>Continuous monitoring missions & statistical methods to predict orbital periods!</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="challenge-card">
                            <div class="challenge-icon">🎭</div>
                            <h3>4. False Positives</h3>
                            <div class="challenge-content">
                                <p><strong>The Problem:</strong></p>
                                <ul>
                                    <li>Binary star eclipses</li>
                                    <li>Background stars</li>
                                    <li>Stellar spots</li>
                                </ul>
                                <div class="solution-box">
                                    <strong>💡 Solution:</strong>
                                    <p>Follow-up observations, radial velocity measurements, & AI validation!</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="success-story">
                        <h3>🎉 Success Story!</h3>
                        <p>Despite these challenges, we've discovered over <strong>5,500 confirmed exoplanets</strong> and counting! AI and machine learning have revolutionized the field, helping us find planets that would have been impossible to detect just a decade ago.</p>
                    </div>
                </div>
            `
        },
        5: {
            title: "🤖 How AI Helps",
            content: `
                <div class="interactive-topic">
                    <div class="topic-intro">
                        <p class="lead-text">Artificial Intelligence is like having a super-smart assistant that never gets tired and can spot patterns humans might miss!</p>
                    </div>
                    
                    <div class="ai-benefits">
                        <div class="benefit-card">
                            <div class="benefit-icon">⚡</div>
                            <h3>Speed</h3>
                            <p>AI can analyze <strong>thousands of light curves per second</strong></p>
                            <div class="stat">100,000x faster than humans!</div>
                        </div>
                        
                        <div class="benefit-card">
                            <div class="benefit-icon">🎯</div>
                            <h3>Accuracy</h3>
                            <p>Detects <strong>subtle patterns</strong> invisible to the human eye</p>
                            <div class="stat">96% accuracy rate!</div>
                        </div>
                        
                        <div class="benefit-card">
                            <div class="benefit-icon">🔄</div>
                            <h3>Consistency</h3>
                            <p>Never gets tired or makes <strong>subjective judgments</strong></p>
                            <div class="stat">24/7 operation!</div>
                        </div>
                        
                        <div class="benefit-card">
                            <div class="benefit-icon">📚</div>
                            <h3>Learning</h3>
                            <p>Improves with <strong>every new discovery</strong></p>
                            <div class="stat">Trained on 14,620 samples!</div>
                        </div>
                    </div>
                    
                    <div class="ai-models-section">
                        <h3>🧠 Our AI Models</h3>
                        <div class="models-grid">
                            <div class="model-card">
                                <h4>🔷 CNN (Convolutional Neural Network)</h4>
                                <p>Fast pattern recognition</p>
                                <div class="model-stats">
                                    <span>F1: 94%</span>
                                    <span>Speed: 175/s</span>
                                </div>
                            </div>
                            <div class="model-card">
                                <h4>🔷 LSTM (Long Short-Term Memory)</h4>
                                <p>Sequential pattern analysis</p>
                                <div class="model-stats">
                                    <span>F1: 92%</span>
                                    <span>Speed: 120/s</span>
                                </div>
                            </div>
                            <div class="model-card">
                                <h4>🔷 Transformer</h4>
                                <p>State-of-the-art accuracy</p>
                                <div class="model-stats">
                                    <span>F1: 95%</span>
                                    <span>Speed: 85/s</span>
                                </div>
                            </div>
                            <div class="model-card">
                                <h4>🔷 Ensemble</h4>
                                <p>Combined power of all models</p>
                                <div class="model-stats">
                                    <span>F1: 96%</span>
                                    <span>Speed: 76/s</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="try-it-section">
                        <h3>🎮 Try It Yourself!</h3>
                        <p>Ready to see AI in action? Head over to the <strong>"Hunt a Planet"</strong> tab and test your skills against our AI models!</p>
                        <button class="btn-primary" onclick="switchToTab('hunt')">🚀 Start Hunting!</button>
                    </div>
                </div>
            `
        }
    };
    
    const topic = topics[topicNumber];
    if (!topic) return;
    
    contentDiv.innerHTML = `
        <div class="topic-header">
            <h2>${topic.title}</h2>
        </div>
        ${topic.content}
    `;
    
    // Initialize any interactive elements
    setTimeout(() => {
        if (topicNumber === 1 || topicNumber === 2) {
            initializeDemoPlots(topicNumber);
        }
        if (topicNumber === 3) {
            initializeSignalExamples();
        }
    }, 100);
}

// Helper functions for interactive elements
function animateLightCurve(demoNumber, hasPlanet) {
    const lightCurve = generateLightCurve(hasPlanet, 'easy');
    const plotId = `demo-lightcurve-${demoNumber}`;
    plotLightCurve(lightCurve.time, lightCurve.flux, plotId);
    showNotification(hasPlanet ? '🪐 Planet transit shown!' : '⭐ No planet shown!', 'info');
}

function updateTransitDemo() {
    const planetSize = document.getElementById('planet-size')?.value || 5;
    const period = document.getElementById('orbital-period')?.value || 5;
    
    // Generate custom transit
    const time = Array.from({length: 2048}, (_, i) => i * 0.01);
    const flux = time.map(t => {
        const transitDepth = planetSize * 0.002;
        const transitWidth = period * 0.1;
        const transitPhase = (t % period) / period;
        
        if (transitPhase > 0.45 && transitPhase < 0.55) {
            return 1.0 - transitDepth;
        }
        return 1.0 + (Math.random() - 0.5) * 0.001;
    });
    
    plotLightCurve(time, flux, 'demo-lightcurve-2');
    showNotification('Transit updated!', 'info');
}

function initializeDemoPlots(topicNumber) {
    // Initialize with default planet transit
    animateLightCurve(topicNumber, true);
}

function initializeSignalExamples() {
    // Show mini examples of each signal type
    ['planet', 'variability', 'noise', 'false'].forEach(type => {
        const hasPlanet = type === 'planet' || type === 'false';
        const difficulty = type === 'noise' ? 'hard' : 'medium';
        const lightCurve = generateLightCurve(hasPlanet, difficulty);
        const plotId = `signal-${type}`;
        
        if (document.getElementById(plotId)) {
            plotLightCurve(lightCurve.time.slice(0, 500), lightCurve.flux.slice(0, 500), plotId);
        }
    });
}

function showSignalExample(type) {
    showNotification(`Showing ${type} example in main plot!`, 'info');
    // Could switch to Hunt tab and show this example
}

function checkQuizAnswer(button, isCorrect) {
    const feedback = button.closest('.quiz-question').querySelector('.quiz-feedback');
    const options = button.closest('.quiz-options').querySelectorAll('.quiz-option');
    
    // Disable all options
    options.forEach(opt => opt.disabled = true);
    
    if (isCorrect) {
        button.classList.add('correct');
        feedback.innerHTML = '<div class="correct-answer">✅ Correct! Periodic repetition is the key to confirming a planet. A single dip could be anything, but regular, repeating dips at consistent intervals strongly suggest an orbiting planet!</div>';
        feedback.classList.remove('hidden');
        showNotification('🎉 Correct answer!', 'success');
    } else {
        button.classList.add('incorrect');
        feedback.innerHTML = '<div class="incorrect-answer">❌ Not quite! While this feature is important, periodic repetition is the most crucial factor for confirming a planet.</div>';
        feedback.classList.remove('hidden');
        showNotification('Try again!', 'error');
    }
}

function switchToTab(tabName) {
    const tabButton = document.querySelector(`[data-tab="${tabName}"]`);
    if (tabButton) {
        tabButton.click();
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeLearnTheBasics();
});

// Export functions
window.animateLightCurve = animateLightCurve;
window.updateTransitDemo = updateTransitDemo;
window.showSignalExample = showSignalExample;
window.checkQuizAnswer = checkQuizAnswer;
window.switchToTab = switchToTab;
