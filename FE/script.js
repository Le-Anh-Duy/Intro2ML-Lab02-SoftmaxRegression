const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultsDiv = document.getElementById('results');
const loadingIndicator = document.getElementById('loadingIndicator');

// Canvas drawing state
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize canvas
function initCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    [lastX, lastY] = [e.clientX - rect.left, e.clientY - rect.top];
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    isDrawing = false;
}

// Touch support
function getTouchPos(e) {
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    return {
        x: touch.clientX - rect.left,
        y: touch.clientY - rect.top
    };
}

function handleTouchStart(e) {
    e.preventDefault();
    const pos = getTouchPos(e);
    isDrawing = true;
    [lastX, lastY] = [pos.x, pos.y];
}

function handleTouchMove(e) {
    e.preventDefault();
    if (!isDrawing) return;
    
    const pos = getTouchPos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    
    [lastX, lastY] = [pos.x, pos.y];
}

function handleTouchEnd(e) {
    e.preventDefault();
    isDrawing = false;
}

// Event listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', handleTouchStart);
canvas.addEventListener('touchmove', handleTouchMove);
canvas.addEventListener('touchend', handleTouchEnd);

clearBtn.addEventListener('click', () => {
    initCanvas();
    resultsDiv.innerHTML = '';
});

predictBtn.addEventListener('click', async () => {
    // Get canvas data as base64
    const imageData = canvas.toDataURL('image/png');
    
    // Show loading
    loadingIndicator.classList.remove('hidden');
    resultsDiv.innerHTML = '';
    
    try {
        // Send to backend
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display results
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        resultsDiv.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${error.message}
                <br>Make sure the backend server is running on http://localhost:5000
            </div>
        `;
    } finally {
        loadingIndicator.classList.add('hidden');
    }
});

function displayResults(data) {
    const modelNames = {
        'pixel_model': 'Pixel-based Model',
        'edge_model': 'Edge-based Model',
        'pca_model': 'PCA-based Model'
    };
    
    resultsDiv.innerHTML = '';
    
    for (const [modelKey, modelName] of Object.entries(modelNames)) {
        if (data[modelKey]) {
            const result = data[modelKey];
            const resultDiv = createResultCard(modelName, result);
            resultsDiv.appendChild(resultDiv);
        }
    }
}

function createResultCard(modelName, result) {
    const card = document.createElement('div');
    card.className = 'model-result';
    
    const confidencePercent = (result.confidence * 100).toFixed(1);
    
    card.innerHTML = `
        <h3>
            <span class="model-icon"></span>
            ${modelName}
        </h3>
        <div class="prediction-display">
            <div class="prediction-digit">${result.prediction}</div>
            <div class="prediction-info">
                <div class="confidence">
                    Confidence: <span class="confidence-value">${confidencePercent}%</span>
                </div>
                <p>Predicted digit: <strong>${result.prediction}</strong></p>
            </div>
        </div>
        <div class="probabilities">
            <h4>Probability Distribution</h4>
            <div class="prob-bars">
                ${result.probabilities.map((prob, idx) => `
                    <div class="prob-bar">
                        <div class="prob-label">${idx}</div>
                        <div class="prob-fill-container">
                            <div class="prob-fill" style="height: ${prob * 100}%"></div>
                        </div>
                        <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    return card;
}

// Initialize canvas on load
initCanvas();