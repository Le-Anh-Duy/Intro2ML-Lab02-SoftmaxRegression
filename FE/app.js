// Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// Canvas setup
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let hasDrawn = false;

// Initialize canvas
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'white';
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Drawing event listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch support
canvas.addEventListener('touchstart', handleTouchStart);
canvas.addEventListener('touchmove', handleTouchMove);
canvas.addEventListener('touchend', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    const pos = getMousePos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    hasDrawn = true;
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const pos = getMousePos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
}

function stopDrawing() {
    isDrawing = false;
}

function handleTouchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

function handleTouchMove(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

// Clear canvas
document.getElementById('clearBtn').addEventListener('click', () => {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('results').innerHTML = '';
    document.getElementById('noResults').classList.remove('hidden');
    document.getElementById('preprocessedSection').classList.add('hidden');
    hasDrawn = false;
});

// Load available models
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/models`);
        const data = await response.json();
        
        const modelSelect = document.getElementById('modelSelect');
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            modelSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Predict digit
document.getElementById('predictBtn').addEventListener('click', async () => {
    if (!hasDrawn) {
        alert('Please draw a digit first!');
        return;
    }
    
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsDiv = document.getElementById('results');
    const noResults = document.getElementById('noResults');
    
    // Show loading
    loadingIndicator.classList.remove('hidden');
    resultsDiv.innerHTML = '';
    noResults.classList.add('hidden');
    
    try {
        // Get canvas image data
        const imageData = canvas.toDataURL('image/png');
        
        // Get selected model
        const selectedModel = document.getElementById('modelSelect').value;
        
        // Send to API
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData,
                model: selectedModel
            })
        });
        
        const data = await response.json();
        
        // Hide loading
        loadingIndicator.classList.add('hidden');
        
        if (data.success) {
            // Display preprocessed image
            if (data.preprocessed_image) {
                const preprocessedSection = document.getElementById('preprocessedSection');
                const preprocessedImage = document.getElementById('preprocessedImage');
                preprocessedImage.src = data.preprocessed_image;
                preprocessedSection.classList.remove('hidden');
            }
            
            displayResults(data.predictions);
        } else {
            resultsDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
        }
    } catch (error) {
        loadingIndicator.classList.add('hidden');
        resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
});

function displayResults(predictions) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';
    
    // Sort models by confidence
    const sortedPredictions = Object.entries(predictions).sort((a, b) => 
        (b[1].confidence || 0) - (a[1].confidence || 0)
    );
    
    sortedPredictions.forEach(([modelKey, result]) => {
        if (result.error) {
            const errorCard = document.createElement('div');
            errorCard.className = 'result-card error';
            errorCard.innerHTML = `
                <h3>${result.model_name || modelKey}</h3>
                <p class="error-message">Error: ${result.error}</p>
            `;
            resultsDiv.appendChild(errorCard);
            return;
        }
        
        const card = document.createElement('div');
        card.className = 'result-card';
        
        const confidencePercent = (result.confidence * 100).toFixed(1);
        const confidenceClass = result.confidence > 0.9 ? 'high' : 
                               result.confidence > 0.7 ? 'medium' : 'low';
        
        card.innerHTML = `
            <div class="model-header">
                <h3>${result.model_name}</h3>
                <div class="prediction-badge">
                    <span class="digit">${result.digit}</span>
                    <span class="confidence ${confidenceClass}">${confidencePercent}%</span>
                </div>
            </div>
            
            <div class="visualization-section">
                <h4>Feature Visualization</h4>
                <img src="${result.visualization}" alt="Feature visualization" class="feature-viz">
                <p class="viz-caption">Learned features for digit ${result.digit}</p>
            </div>
            
            <div class="probabilities-section">
                <h4>Class Probabilities</h4>
                <div class="prob-bars">
                    ${createProbabilityBars(result.probabilities, result.digit)}
                </div>
            </div>
        `;
        
        resultsDiv.appendChild(card);
    });
}

function createProbabilityBars(probabilities, predictedDigit) {
    return probabilities.map((prob, idx) => {
        const percent = (prob * 100).toFixed(1);
        const isPredicted = idx === predictedDigit;
        
        return `
            <div class="prob-bar-container ${isPredicted ? 'predicted' : ''}">
                <span class="prob-label">${idx}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar" style="width: ${percent}%"></div>
                </div>
                <span class="prob-value">${percent}%</span>
            </div>
        `;
    }).join('');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    document.getElementById('noResults').classList.remove('hidden');
});
