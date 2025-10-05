// NASA Exoplanet Detection Pipeline - Main Application Logic

// Global State
const appState = {
    currentPage: 'home',
    currentTab: null,
    score: 0,
    attempts: 0,
    sessionHistory: [],
    currentSample: null,
    challengeMode: false
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    initializeTabs();
    loadHomePage();
});

// Navigation System
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = link.dataset.page;
            navigateTo(page);
        });
    });
}

function navigateTo(pageName) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show selected page
    const targetPage = document.getElementById(`${pageName}-page`);
    if (targetPage) {
        targetPage.classList.add('active');
    }
    
    // Update navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.dataset.page === pageName) {
            link.classList.add('active');
        }
    });
    
    appState.currentPage = pageName;
    
    // Load page-specific content
    switch(pageName) {
        case 'home':
            loadHomePage();
            break;
        case 'beginner':
            loadBeginnerMode();
            break;
        case 'research':
            loadResearchMode();
            break;
        case 'comparison':
            loadComparisonPage();
            break;
        case 'about':
            loadAboutPage();
            break;
    }
}

// Tab System
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            const parentPage = button.closest('.page');
            
            // Hide all tab contents in this page
            parentPage.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tab buttons in this page
            parentPage.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Show selected tab content
            const targetTab = parentPage.querySelector(`#${tabName}-tab`);
            if (targetTab) {
                targetTab.classList.add('active');
            }
            
            // Add active class to clicked button
            button.classList.add('active');
            
            appState.currentTab = tabName;
        });
    });
}

// Page Loaders
function loadHomePage() {
    console.log('Home page loaded');
    // Home page is static, no dynamic loading needed
}

function loadBeginnerMode() {
    console.log('Beginner mode loaded');
    // Initialize first sample for Hunt a Planet
    if (appState.currentTab === null || appState.currentTab === 'hunt') {
        newSample();
    }
}

function loadResearchMode() {
    console.log('Research mode loaded');
}

function loadComparisonPage() {
    console.log('Comparison page loaded');
}

function loadAboutPage() {
    console.log('About page loaded');
}

// Utility Functions
function updateScore(points) {
    appState.score += points;
    document.getElementById('score').textContent = appState.score;
}

function updateAttempts() {
    appState.attempts++;
    document.getElementById('attempts').textContent = appState.attempts;
    updateAccuracy();
}

function updateAccuracy() {
    if (appState.attempts > 0) {
        const accuracy = (appState.score / appState.attempts * 100).toFixed(1);
        document.getElementById('accuracy').textContent = `${accuracy}%`;
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 2rem;
        background: ${type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--error)' : 'var(--info)'};
        color: white;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-lg);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Export functions for use in other modules
window.appState = appState;
window.navigateTo = navigateTo;
window.updateScore = updateScore;
window.updateAttempts = updateAttempts;
window.showNotification = showNotification;


// Enhanced Visualizations Page Loader
function loadVisualizationsPage() {
    console.log('Enhanced Visualizations page loaded');
    
    // Load metric cards
    const metricsContainer = document.getElementById('metric-cards-container');
    if (metricsContainer) {
        metricsContainer.innerHTML = `
            ${createMetricCard('Best F1 Score', '97.5%', '+12.5%', '🥇')}
            ${createMetricCard('Fastest Model', 'CNN', '175/s', '⚡')}
            ${createMetricCard('Most Efficient', 'Transformer', '2.55', '🎯')}
            ${createMetricCard('Total Models', '5', 'Ready', '🤖')}
        `;
    }
    
    // Load performance dashboard
    setTimeout(() => {
        if (document.getElementById('performance-dashboard-plot')) {
            createPerformanceDashboard('performance-dashboard-plot');
        }
    }, 100);
    
    // Load architecture comparison
    setTimeout(() => {
        if (document.getElementById('architecture-plot')) {
            createArchitectureComparison('architecture-plot');
        }
    }, 100);
    
    // Load efficiency analysis
    setTimeout(() => {
        if (document.getElementById('efficiency-plot')) {
            createEfficiencyScatterMatrix('efficiency-plot');
        }
    }, 100);
}

// Light Curve Functions
function loadSampleLightCurve(type) {
    const hasPlanet = type === 'planet';
    const { time, flux } = generateLightCurve(hasPlanet, 'medium');
    
    const prediction = {
        probability: hasPlanet ? 0.92 : 0.15,
        confidence: 0.88
    };
    
    createLightCurveAnalyzer('lightcurve-plot', time, flux, prediction);
    showNotification(`Loaded ${type} example`, 'success');
}

function generateRandomLightCurve() {
    const hasPlanet = Math.random() > 0.5;
    const difficulty = ['easy', 'medium', 'hard'][Math.floor(Math.random() * 3)];
    const { time, flux } = generateLightCurve(hasPlanet, difficulty);
    
    const prediction = {
        probability: hasPlanet ? (0.7 + Math.random() * 0.25) : (0.1 + Math.random() * 0.3),
        confidence: 0.8 + Math.random() * 0.15
    };
    
    createLightCurveAnalyzer('lightcurve-plot', time, flux, prediction);
    showNotification('Generated random light curve', 'info');
}

// Update navigateTo function to include visualizations page
const originalNavigateTo = window.navigateTo;
window.navigateTo = function(pageName) {
    originalNavigateTo(pageName);
    
    if (pageName === 'visualizations') {
        loadVisualizationsPage();
    }
};

// Export functions
window.loadSampleLightCurve = loadSampleLightCurve;
window.generateRandomLightCurve = generateRandomLightCurve;
window.loadVisualizationsPage = loadVisualizationsPage;
