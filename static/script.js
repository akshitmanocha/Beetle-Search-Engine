// API endpoint
const API_URL = 'http://localhost:8000';

// DOM elements
const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');
const searchMethod = document.getElementById('searchMethod');
const topK = document.getElementById('topK');
const rerankK = document.getElementById('rerankK');
const rerankerEnabled = document.getElementById('rerankerEnabled');

const loadingSpinner = document.getElementById('loadingSpinner');
const resultsContainer = document.getElementById('resultsContainer');
const resultsTitle = document.getElementById('resultsTitle');
const resultsInfo = document.getElementById('resultsInfo');
const resultsList = document.getElementById('resultsList');
const errorMessage = document.getElementById('errorMessage');

// Load config on page load
async function loadConfig() {
    try {
        const response = await fetch(`${API_URL}/config`);
        if (response.ok) {
            const config = await response.json();
            if (config.search) {
                topK.value = config.search.top_k || 20;
                rerankK.value = config.search.rerank_k || 5;
                searchMethod.value = config.search.method || 'hybrid';
                rerankerEnabled.checked = config.search.reranker || false;
            }
        }
    } catch (error) {
        console.log('Could not load config, using defaults');
    }
}

// Perform search
async function performSearch() {
    const query = searchInput.value.trim();

    if (!query) {
        showError('Please enter a search query');
        return;
    }

    // Hide previous results/errors
    hideAll();

    // Show loading spinner
    loadingSpinner.style.display = 'block';

    const requestBody = {
        query: query,
        top_k: parseInt(topK.value),
        rerank_k: parseInt(rerankK.value),
        search_method: searchMethod.value,
        reranker_enabled: rerankerEnabled.checked
    };

    try {
        const response = await fetch(`${API_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Search failed');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Search error:', error);
        showError(`Search failed: ${error.message}`);
    } finally {
        loadingSpinner.style.display = 'none';
    }
}

// Display search results
function displayResults(data) {
    // Update results header
    resultsTitle.textContent = `Results for "${data.query}"`;

    const methodText = data.search_method.toUpperCase();
    const rerankText = data.reranker_enabled ? ' with Reranking' : '';
    resultsInfo.innerHTML = `
        <strong>${data.total_results}</strong> results found using
        <strong>${methodText}</strong>${rerankText}
    `;

    // Clear previous results
    resultsList.innerHTML = '';

    // Display each result
    if (data.results.length === 0) {
        resultsList.innerHTML = '<p style="text-align: center; color: #666; padding: 40px;">No results found</p>';
    } else {
        data.results.forEach((result, index) => {
            const resultItem = createResultItem(result, index + 1, data.reranker_enabled);
            resultsList.appendChild(resultItem);
        });
    }

    // Show results container
    resultsContainer.style.display = 'block';
}

// Create a result item element
function createResultItem(result, number, hasReranking) {
    const item = document.createElement('div');
    item.className = 'result-item';

    const title = document.createElement('div');
    title.className = 'result-title';
    title.innerHTML = `
        <span class="result-number">${number}</span>
        ${escapeHtml(result.title || 'Untitled')}
    `;

    const link = document.createElement('a');
    link.className = 'result-link';
    link.href = result.url || '#';
    link.target = '_blank';
    link.textContent = result.url || 'No URL';

    item.appendChild(title);
    item.appendChild(link);

    // Display score if available
    if (hasReranking && result.rerank_score !== undefined) {
        const score = document.createElement('span');
        score.className = 'result-score';
        score.textContent = `Score: ${result.rerank_score.toFixed(4)}`;
        item.appendChild(score);
    }

    // Display description if available
    if (result.content) {
        const description = document.createElement('div');
        description.className = 'result-description';
        const truncatedContent = result.content.substring(0, 300) + (result.content.length > 300 ? '...' : '');
        description.textContent = truncatedContent;
        item.appendChild(description);
    }

    return item;
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 5000);
}

// Hide all result containers
function hideAll() {
    resultsContainer.style.display = 'none';
    errorMessage.style.display = 'none';
    loadingSpinner.style.display = 'none';
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Event listeners
searchButton.addEventListener('click', performSearch);

searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

// Load config on page load
loadConfig();
