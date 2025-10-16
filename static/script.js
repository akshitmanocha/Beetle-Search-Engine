// API endpoint
const API_URL = 'http://localhost:8000';

// DOM elements
const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');

const loadingSpinner = document.getElementById('loadingSpinner');
const resultsContainer = document.getElementById('resultsContainer');
const resultsTitle = document.getElementById('resultsTitle');
const resultsInfo = document.getElementById('resultsInfo');
const resultsList = document.getElementById('resultsList');
const errorMessage = document.getElementById('errorMessage');
const errorMessageWrapper = document.getElementById('errorMessageWrapper');

// Search configuration from params.yaml
let searchConfig = {
    top_k: 20,
    rerank_k: 5,
    method: 'hybrid',
    reranker: false
};

// Load config from backend on page load
async function loadConfig() {
    try {
        const response = await fetch(`${API_URL}/config`);
        if (response.ok) {
            const config = await response.json();
            if (config.search) {
                searchConfig.top_k = config.search.top_k || 20;
                searchConfig.rerank_k = config.search.rerank_k || 5;
                searchConfig.method = config.search.method || 'hybrid';
                searchConfig.reranker = config.search.reranker || false;
                console.log('Loaded search configuration:', searchConfig);
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
        top_k: searchConfig.top_k,
        rerank_k: searchConfig.rerank_k,
        search_method: searchConfig.method,
        reranker_enabled: searchConfig.reranker
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
    item.className = 'py-4 border-b border-gray-200 last:border-b-0 hover:bg-gray-50 transition-colors';

    // Create clickable title with embedded link
    const titleLink = document.createElement('a');
    titleLink.className = 'block text-xl font-normal text-blue-700 hover:underline mb-1 visited:text-purple-700';
    titleLink.href = result.url || '#';
    titleLink.target = '_blank';
    titleLink.textContent = result.title || 'Untitled';

    item.appendChild(titleLink);

    // Display score if available
    if (hasReranking && result.rerank_score !== undefined) {
        const score = document.createElement('span');
        score.className = 'inline-block bg-blue-50 text-blue-700 text-xs font-medium px-2.5 py-1 rounded-full mb-2';
        score.textContent = `Score: ${result.rerank_score.toFixed(4)}`;
        item.appendChild(score);
    }

    // Display body content preview
    const bodyContent = result.body_text || result.content || '';
    if (bodyContent) {
        const description = document.createElement('div');
        description.className = 'text-sm text-gray-700 leading-relaxed mt-1';
        const truncatedContent = bodyContent.substring(0, 350) + (bodyContent.length > 350 ? '...' : '');
        description.textContent = truncatedContent;
        item.appendChild(description);
    }

    return item;
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessageWrapper.style.display = 'block';
    setTimeout(() => {
        errorMessageWrapper.style.display = 'none';
    }, 5000);
}

// Hide all result containers
function hideAll() {
    resultsContainer.style.display = 'none';
    errorMessageWrapper.style.display = 'none';
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
