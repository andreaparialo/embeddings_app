// ================================================
// Results Manager Module
// ================================================

export class ResultsManager {
    constructor() {
        this.resultCardTemplate = null;
    }

    showResults(results, containerId, countId) {
        const container = document.getElementById(containerId);
        const grid = document.getElementById(containerId.replace('-container', '-grid'));
        const countBadge = document.getElementById(countId);
        
        if (!container || !grid || !countBadge) {
            console.error('Results container elements not found');
            return;
        }
        
        // Update count
        countBadge.textContent = `${results.length} results`;
        
        // Clear existing results
        grid.innerHTML = '';
        
        // Add new results
        results.forEach(result => {
            const card = this.createResultCard(result);
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
    }

    createResultCard(result) {
        const card = document.createElement('div');
        card.className = 'result-card';
        
        // Build image HTML
        let imageHtml = '';
        if (result.image_path) {
            const imgSrc = `/${result.image_path}`;
            imageHtml = `
                <img src="${imgSrc}" 
                     class="result-image" 
                     alt="Product Image" 
                     loading="lazy"
                     onerror="this.onerror=null; this.src='/static/images/no-image.svg';">
            `;
        } else {
            imageHtml = `
                <div class="result-image no-image">
                    <i class="fas fa-image"></i>
                    <span>No Image</span>
                </div>
            `;
        }
        
        // Build similarity badge if applicable
        let similarityHtml = '';
        if (result.similarity_score !== undefined) {
            const percentage = ((1 - result.similarity_score) * 100).toFixed(1);
            similarityHtml = `<div class="similarity-badge">${percentage}%</div>`;
        }
        
        // Build card HTML
        card.innerHTML = `
            ${similarityHtml}
            ${imageHtml}
            <div class="result-content">
                <h6 class="result-sku">${this.escapeHtml(result.SKU_COD || 'N/A')}</h6>
                <div class="result-details">
                    ${this.buildDetailsHtml(result)}
                </div>
                ${result.filename_root ? `
                    <div class="result-meta">
                        <small class="text-muted">File: ${this.escapeHtml(result.filename_root)}</small>
                    </div>
                ` : ''}
            </div>
        `;
        
        // Add click handler for detailed view
        card.addEventListener('click', () => {
            this.showDetailedView(result);
        });
        
        return card;
    }

    buildDetailsHtml(result) {
        const details = [
            { label: 'Brand', value: result.BRAND_DES },
            { label: 'Cluster', value: result.BRAND_CLUSTER },
            { label: 'Gender', value: result.USERGENDER_DES },
            { label: 'Color', value: result.COLOR_FAMILY_1_DES, extra: result.COLOR },
            { label: 'Shape', value: result.SHAPE_SEMI_GROUPED },
            { label: 'Price', value: result.ACT_SKU_PRICE_VAL }
        ];
        
        return details
            .filter(d => d.value)
            .map(d => `
                <div class="detail-item">
                    <strong>${d.label}:</strong> 
                    ${this.escapeHtml(d.value)}
                    ${d.extra ? ` (${this.escapeHtml(d.extra)})` : ''}
                </div>
            `)
            .join('');
    }

    showDetailedView(result) {
        // TODO: Implement modal or side panel for detailed product view
        console.log('Show detailed view for:', result);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    clearResults(containerId) {
        const container = document.getElementById(containerId);
        const grid = document.getElementById(containerId.replace('-container', '-grid'));
        
        if (container) {
            container.style.display = 'none';
        }
        
        if (grid) {
            grid.innerHTML = '';
        }
    }
} 