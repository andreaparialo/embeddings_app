// ================================================
// Batch Viewer Manager Module - Redesigned
// ================================================

import { ProductCardComponent } from './productCard.js?v=5';

console.log('BatchViewer module loaded');

export class BatchViewerManager {
    constructor() {
        console.log('BatchViewer constructor called');
        this.currentIndex = 0;
        this.batchResults = {};
        this.skuList = [];
        this.currentInputSKU = null;
        this.sessionId = null;
        this.metadata = {};
        this.productCard = new ProductCardComponent();
    }

    init() {
        console.log('🚀 BatchViewer: Initializing...');
        
        // Wait for DOM to be ready if needed
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.doInit());
        } else {
            this.doInit();
        }
    }
    
    doInit() {
        console.log('📱 BatchViewer: DOM ready, attaching handlers...');
        this.attachUploadHandler();
        this.attachKeyboardShortcuts();
        console.log('✅ BatchViewer: Initialized successfully');
    }

    attachUploadHandler() {
        console.log('🚀 BatchViewer: attachUploadHandler called');
        console.log('📍 Current location:', window.location.href);
        console.log('📄 Document ready state:', document.readyState);
        
        const uploadButton = document.getElementById('upload-excel-btn');
        console.log('🔍 BatchViewer: Looking for upload button...', uploadButton ? 'FOUND' : 'NOT FOUND');
        
        if (uploadButton) {
            console.log('✅ Batch viewer: Attaching upload handler to button');
            console.log('🎯 Button element:', uploadButton);
            console.log('🏷️ Button innerHTML:', uploadButton.innerHTML);
            
            // Test click directly on the button
            uploadButton.addEventListener('click', (e) => {
                console.log('🎯 DIRECT CLICK HANDLER TRIGGERED!', e);
                console.log('📤 Upload button clicked - processing...');
                this.handleExcelUpload(e);
            });
            
            // Also add a simple test handler
            uploadButton.addEventListener('click', () => {
                console.log('🟢 SIMPLE TEST HANDLER WORKS!');
            });
            
            console.log('🎯 Event listeners attached successfully');
            
            // Test if the button is clickable
            console.log('🧪 Testing button visibility and interactivity...');
            const styles = window.getComputedStyle(uploadButton);
            console.log('👁️ Button visibility:', styles.visibility);
            console.log('🖱️ Button pointer-events:', styles.pointerEvents);
            console.log('📏 Button display:', styles.display);
            
        } else {
            console.warn('⚠️ Batch viewer: Upload button not found! Retrying in 1 second...');
            console.log('🔍 All buttons on page:', document.querySelectorAll('button'));
            console.log('🔍 All elements with upload:', document.querySelectorAll('[id*="upload"]'));
            
            // Prevent infinite recursion by limiting retries
            this.uploadHandlerRetries = (this.uploadHandlerRetries || 0) + 1;
            if (this.uploadHandlerRetries < 10) {
                setTimeout(() => this.attachUploadHandler(), 1000);
            } else {
                console.error('❌ BatchViewer: Failed to find upload button after 10 retries');
            }
        }
    }

    attachKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Only handle shortcuts when batch viewer is active
            const batchViewerView = document.getElementById('batch-viewer-view');
            if (!batchViewerView || !batchViewerView.classList.contains('active')) {
                return;
            }

            switch (e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.previousSKU();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.nextSKU();
                    break;
                case 'Home':
                    e.preventDefault();
                    this.goToSKU(0);
                    break;
                case 'End':
                    e.preventDefault();
                    this.goToSKU(this.skuList.length - 1);
                    break;
            }
        });
    }

    async handleExcelUpload(e) {
        console.log('Batch viewer: Upload button clicked, processing upload...');
        
        const fileInput = document.getElementById('excel-results-file');
        
        if (!fileInput || !fileInput.files.length) {
            this.showNotification('Please select an Excel file to upload.', 'error');
            return;
        }
        
        console.log('Batch viewer: File selected:', fileInput.files[0].name);
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        try {
            this.setUploadButtonLoading(true);
            
            console.log('Batch viewer: Making fetch request to /api/upload-excel-results');
            const response = await fetch('/api/upload-excel-results', {
                method: 'POST',
                body: formData
            });
            console.log('Batch viewer: Response received:', response.status, response.statusText);
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || 'Upload failed');
            }
            
            this.showNotification(
                `${result.message}\\nFound ${result.total_results} results for ${result.total_skus} SKUs.`,
                'success'
            );
            
            await this.loadBatchResults(result.session_id);
            
            // Reset form
            fileInput.value = '';
            
        } catch (error) {
            console.error('Excel upload error:', error);
            this.showNotification(`Error uploading Excel file: ${error.message}`, 'error');
        } finally {
            this.setUploadButtonLoading(false);
        }
    }

    setUploadButtonLoading(isLoading) {
        const submitButton = document.getElementById('upload-excel-btn');
        if (submitButton) {
            if (isLoading) {
                submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                submitButton.disabled = true;
            } else {
                submitButton.innerHTML = '<i class="fas fa-upload"></i> Load Results for Viewing';
                submitButton.disabled = false;
            }
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${type === 'error' ? 'fa-exclamation-circle' : 'fa-check-circle'}"></i>
                <span>${message}</span>
                <button class="notification-close"><i class="fas fa-times"></i></button>
            </div>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);

        // Manual close
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });
    }

    async loadBatchResults(sessionId) {
        this.sessionId = sessionId;
        
        try {
            const response = await fetch(`/api/batch-results/${sessionId}`);
            if (!response.ok) {
                throw new Error(`Failed to load batch results: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.batchResults = data.results || {};
            this.metadata = data.metadata || {};
            
            // Convert results object to array format for navigation
            this.skuList = Object.keys(this.batchResults);
            this.currentIndex = 0;
            this.currentInputSKU = this.skuList[0] || null;
            
            this.renderBatchViewer();
            
        } catch (error) {
            console.error('Error loading batch results:', error);
            this.showNotification(`Error loading batch results: ${error.message}`, 'error');
        }
    }

    renderBatchViewer() {
        const container = document.getElementById('batch-viewer-content');
        if (!container) return;
        
        // Hide upload section and show results
        const uploadSection = document.getElementById('batch-viewer-upload');
        const resultsSection = document.getElementById('batch-viewer-results');
        
        if (uploadSection) uploadSection.style.display = 'none';
        if (resultsSection) resultsSection.style.display = 'block';
        
        if (!this.currentInputSKU || !this.batchResults[this.currentInputSKU]) {
            resultsSection.innerHTML = '<div class="empty-state"><i class="fas fa-search"></i><p>No results to display.</p></div>';
            return;
        }
        
        const currentResults = this.batchResults[this.currentInputSKU];
        const totalSkus = this.skuList.length;
        
        // Create the new layout
        resultsSection.innerHTML = `
            <div class="batch-viewer-layout">
                <!-- Header -->
                <div class="batch-viewer-header">
                    <div class="header-content">
                        <div class="header-title">
                            <h3>Batch Results Viewer</h3>
                            <div class="batch-stats">
                                <span class="stat-badge primary">SKU ${this.currentIndex + 1} of ${totalSkus}</span>
                                <span class="stat-badge info">${currentResults.length} similar items</span>
                            </div>
                        </div>
                        <div class="header-actions">
                            <button class="btn btn-outline" onclick="window.visionSearchApp.batchViewer.showUploadSection()">
                                <i class="fas fa-upload"></i> Upload New File
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Navigation -->
                <div class="batch-navigation">
                    <div class="nav-controls">
                        <button class="nav-btn nav-btn-prev" 
                                onclick="window.visionSearchApp.batchViewer.previousSKU()" 
                                ${this.currentIndex === 0 ? 'disabled' : ''}>
                            <i class="fas fa-chevron-left"></i>
                            <span>Previous</span>
                        </button>
                        
                        <div class="nav-selector">
                            <select class="nav-select" onchange="window.visionSearchApp.batchViewer.goToSKU(this.value)">
                                ${this.skuList.map((sku, index) => 
                                    `<option value="${index}" ${index === this.currentIndex ? 'selected' : ''}>${sku}</option>`
                                ).join('')}
                            </select>
                            <div class="nav-progress">
                                <div class="progress-bar" style="width: ${((this.currentIndex + 1) / totalSkus) * 100}%"></div>
                            </div>
                        </div>
                        
                        <button class="nav-btn nav-btn-next" 
                                onclick="window.visionSearchApp.batchViewer.nextSKU()" 
                                ${this.currentIndex === totalSkus - 1 ? 'disabled' : ''}>
                            <span>Next</span>
                            <i class="fas fa-chevron-right"></i>
                        </button>
                    </div>
                    <div class="keyboard-hint">
                        <small><i class="fas fa-keyboard"></i> Use ← → arrow keys to navigate</small>
                    </div>
                </div>

                <!-- Main Content Area -->
                <div class="batch-content">
                    <!-- Source Product Panel -->
                    <div class="source-panel">
                        <div class="panel-header">
                            <h4><i class="fas fa-search"></i> Source Product</h4>
                            <div class="source-sku">${this.currentInputSKU}</div>
                        </div>
                        <div class="panel-content" id="source-product-container">
                            ${this.renderSourceProduct(currentResults[0])}
                        </div>
                    </div>

                    <!-- Results Panel -->
                    <div class="results-panel">
                        <div class="panel-header">
                            <h4><i class="fas fa-th-large"></i> Similar Products</h4>
                            <div class="results-count">${currentResults.length} found</div>
                        </div>
                        <div class="panel-content scrollable" id="results-container">
                            <div class="results-grid" id="batch-results-grid">
                                <!-- Results will be rendered here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Render the results grid after HTML is inserted
        setTimeout(() => {
            this.renderResultsToDOM(currentResults);
            // Preload some images for better performance
            this.productCard.preloadImages(currentResults.slice(0, 10));
        }, 0);
    }

    renderSourceProduct(firstResult) {
        if (!firstResult) {
            return '<div class="empty-state"><i class="fas fa-image"></i><p>No source product information available.</p></div>';
        }
        
        // Extract source product data
        const sourceProduct = {};
        Object.keys(firstResult).forEach(key => {
            if (key.startsWith('Source_') || key.startsWith('Input_')) {
                const newKey = key.replace(/^(Source_|Input_)/, '');
                sourceProduct[newKey] = firstResult[key];
            }
        });
        
        // Get the filename root for image loading
        const filenameRoot = sourceProduct.filename_root || sourceProduct.Similar_filename_root || '';
        const sku = sourceProduct.SKU_COD || sourceProduct.SKU || firstResult.Input_SKU || 'Unknown';
        
        // Create custom layout for source product
        const html = `
            <div class="source-product-layout">
                <div class="source-image-section">
                    <div class="product-image-container" id="source-image-${sku}">
                        <div class="product-image-loading">
                            <i class="fas fa-spinner fa-spin"></i>
                            <span>Loading...</span>
                        </div>
                    </div>
                </div>
                
                <div class="source-details-section">
                    <div class="source-details-grid">
                        ${this.renderSourceDetails(sourceProduct)}
                    </div>
                </div>
            </div>
        `;
        
        // Load image after rendering
        setTimeout(() => {
            if (filenameRoot) {
                this.productCard.loadCardImage(
                    { querySelector: () => document.getElementById(`source-image-${sku}`) },
                    { filename_root: filenameRoot }
                );
            } else {
                const container = document.getElementById(`source-image-${sku}`);
                if (container) {
                    container.innerHTML = `
                        <div class="product-image-placeholder">
                            <i class="fas fa-image"></i>
                            <span>No image</span>
                        </div>
                    `;
                }
            }
        }, 100);
        
        return html;
    }
    
    renderSourceDetails(product) {
        const priorityFields = [
            { key: 'SKU_COD', label: 'SKU' },
            { key: 'MODEL_COD', label: 'Model' },
            { key: 'BRAND_DES', label: 'Brand' },
            { key: 'USERGENDER_DES', label: 'Gender' },
            { key: 'MACRO_SHAPE_AWS', label: 'Shape' },
            { key: 'GRANULAR_SHAPE_AWS', label: 'Detailed Shape' },
            { key: 'COLOR_FAMILY_1_DES', label: 'Color Family' },
            { key: 'COLOR', label: 'Color Code' },
            { key: 'RIM_TYPE_DES', label: 'Rim Type' },
            { key: 'CTM_FIRST_FRONT_MATERIAL_DES', label: 'Material' },
            { key: 'MD_SKU_STATUS_COD', label: 'Status' },
            { key: 'PRODUCT_TYPE_COD', label: 'Product Type' },
            { key: 'ACT_SKU_PRICE_VAL', label: 'Price' }
        ];
        
        const detailsHtml = [];
        
        // Add priority fields in order
        for (const field of priorityFields) {
            const value = product[field.key];
            if (value && value !== '' && value !== 'null') {
                detailsHtml.push(`
                    <div class="source-detail-item">
                        <span class="source-detail-label">${field.label}:</span>
                        <span class="source-detail-value">${this.formatValue(field.key, value)}</span>
                    </div>
                `);
            }
        }
        
        // Add any other fields not in priority list
        const addedKeys = priorityFields.map(f => f.key);
        Object.keys(product).forEach(key => {
            if (!addedKeys.includes(key) && 
                !['filename_root', 'similarity_score', 'Similarity_Score'].includes(key) &&
                product[key] && product[key] !== '' && product[key] !== 'null') {
                const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                detailsHtml.push(`
                    <div class="source-detail-item">
                        <span class="source-detail-label">${label}:</span>
                        <span class="source-detail-value">${this.formatValue(key, product[key])}</span>
                    </div>
                `);
            }
        });
        
        return detailsHtml.join('');
    }
    
    formatValue(key, value) {
        if (key.includes('PRICE') && !isNaN(parseFloat(value))) {
            return `€${parseFloat(value).toFixed(2)}`;
        }
        return value;
    }

    renderResultsToDOM(results) {
        const gridContainer = document.getElementById('batch-results-grid');
        if (!gridContainer) return;
        
        // Clear existing content
        gridContainer.innerHTML = '';
        
        if (!results || results.length === 0) {
            gridContainer.innerHTML = '<div class="empty-state"><i class="fas fa-search"></i><p>No similar products found.</p></div>';
            return;
        }
        
        // Create and append cards
        results.forEach((result, index) => {
            const card = this.productCard.createCard(result, {
                showSimilarity: true,
                size: 'normal',
                layout: 'vertical',
                clickable: true,
                showDetails: true,
                maxDetails: 4
            });
            
            card.dataset.resultIndex = index;
            card.classList.add('result-card');
            
            // Append the actual DOM element directly
            gridContainer.appendChild(card);
        });
    }

    showUploadSection() {
        const uploadSection = document.getElementById('batch-viewer-upload');
        const resultsSection = document.getElementById('batch-viewer-results');
        
        if (uploadSection) uploadSection.style.display = 'block';
        if (resultsSection) resultsSection.style.display = 'none';
        
        // Clear current state
        this.batchResults = {};
        this.skuList = [];
        this.currentIndex = 0;
        this.currentInputSKU = null;
        this.sessionId = null;
    }

    previousSKU() {
        if (this.currentIndex > 0) {
            this.currentIndex--;
            this.currentInputSKU = this.skuList[this.currentIndex];
            this.renderBatchViewer();
        }
    }

    nextSKU() {
        if (this.currentIndex < this.skuList.length - 1) {
            this.currentIndex++;
            this.currentInputSKU = this.skuList[this.currentIndex];
            this.renderBatchViewer();
        }
    }

    goToSKU(index) {
        const idx = parseInt(index);
        if (idx >= 0 && idx < this.skuList.length) {
            this.currentIndex = idx;
            this.currentInputSKU = this.skuList[this.currentIndex];
            this.renderBatchViewer();
        }
    }

    async handleBatchSearchComplete(sessionId) {
        this.showNotification('Batch search completed! Loading results...', 'success');
        
        // Switch to batch viewer
        const navigation = window.visionSearchApp?.navigation;
        if (navigation) {
            navigation.showView('batch-viewer');
        }
        
        // Load the results
        await this.loadBatchResults(sessionId);
    }

    // Public API methods
    getCurrentSKU() {
        return this.currentInputSKU;
    }

    getCurrentResults() {
        return this.batchResults[this.currentInputSKU] || [];
    }

    getTotalSKUs() {
        return this.skuList.length;
    }

    getCurrentIndex() {
        return this.currentIndex;
    }
} 