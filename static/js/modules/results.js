// ================================================
// Results Manager Module
// ================================================

import { ProductCardComponent } from './productCard.js?v=5';

export class ResultsManager {
    constructor() {
        this.productCard = new ProductCardComponent();
    }

    showResults(results, containerId, countId) {
        const container = document.getElementById(containerId);
        const grid = document.getElementById(containerId.replace('-container', '-grid'));
        const countBadge = document.getElementById(countId);
        
        if (!container || !grid || !countBadge) {
            console.error('Results container elements not found');
            return;
        }
        
        // Make sure the container is visible
        container.style.display = 'block';
        
        // Update count
        countBadge.textContent = `${results.length} results`;
        
        // Clear existing results
        grid.innerHTML = '';
        
        // Sort results by similarity score if available (higher = better)
        const sortedResults = results.sort((a, b) => {
            const aScore = a.similarity_score !== undefined ? (1 - a.similarity_score) : 0;
            const bScore = b.similarity_score !== undefined ? (1 - b.similarity_score) : 0;
            return bScore - aScore; // Descending order (best first)
        });
        
        // Add new results using ProductCardComponent
        sortedResults.forEach((result, index) => {
            const card = this.productCard.createCard(result, {
                showSimilarity: result.similarity_score !== undefined,
                size: 'normal',
                layout: 'vertical',
                clickable: true,
                showDetails: true,
                maxDetails: 4
            });
            
            card.dataset.resultIndex = index;
            card.classList.add('result-card');
            
            // Append the actual DOM element directly
            grid.appendChild(card);
        });
        
        // Show container
        container.style.display = 'block';
        
        // Smooth scroll to results
        setTimeout(() => {
            container.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'nearest' 
            });
        }, 100);
        
        // Hide loading
        if (window.visionSearchApp) {
            window.visionSearchApp.hideLoading();
        }
        
        // Preload some images for better performance
        this.productCard.preloadImages(sortedResults.slice(0, 10));
    }
} 