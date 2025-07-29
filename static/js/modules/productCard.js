// ================================================
// Product Card Component Module
// ================================================

export class ProductCardComponent {
    constructor() {
        this.loadingQueue = new Set();
        this.imageCache = new Map();
    }

    /**
     * Create a product card element
     * @param {Object} product - Product data
     * @param {Object} options - Display options
     * @returns {HTMLElement} Product card element
     */
    createCard(product, options = {}) {
        const {
            showSimilarity = false,
            size = 'normal', // 'small', 'normal', 'large'
            layout = 'vertical', // 'vertical', 'horizontal'
            clickable = true,
            showDetails = true,
            maxDetails = 5
        } = options;

        const card = document.createElement('div');
        card.className = `product-card ${size} ${layout} ${clickable ? 'clickable' : ''}`;
        card.dataset.sku = product.SKU_COD || product.Similar_SKU || product.sku || '';

        // Create card content
        card.innerHTML = this.generateCardHTML(product, {
            showSimilarity,
            showDetails,
            maxDetails,
            layout
        });

        // Add event listeners
        if (clickable) {
            this.attachCardEvents(card, product);
        }

        // Load image asynchronously
        this.loadCardImage(card, product);

        return card;
    }

    /**
     * Generate card HTML structure
     */
    generateCardHTML(product, options) {
        const { showSimilarity, showDetails, maxDetails, layout } = options;
        const similarity = this.getSimilarityScore(product);
        const sku = this.getSKU(product);
        const filenameRoot = this.getFilenameRoot(product);

        return `
            <div class="product-image-container">
                <div class="product-image-loading">
                    <i class="fas fa-image"></i>
                    <span>Loading...</span>
                </div>
                ${showSimilarity && similarity !== null ? `
                    <div class="similarity-badge">
                        ${(similarity * 100).toFixed(1)}%
                    </div>
                ` : ''}
                <div class="product-overlay">
                    <button class="btn-view-details" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                </div>
            </div>
            <div class="product-content">
                <div class="product-header">
                    <h6 class="product-sku">${sku}</h6>
                    ${similarity !== null ? `
                        <span class="similarity-score">${(similarity * 100).toFixed(1)}%</span>
                    ` : ''}
                </div>
                ${showDetails ? this.generateProductDetails(product, maxDetails) : ''}
            </div>
        `;
    }

    /**
     * Generate product details section
     */
    generateProductDetails(product, maxDetails) {
        const details = this.extractProductDetails(product, maxDetails);
        
        return `
            <div class="product-details">
                ${details.map(({ label, value }) => `
                    <div class="detail-row">
                        <span class="detail-label">${label}:</span>
                        <span class="detail-value">${value}</span>
                    </div>
                `).join('')}
            </div>
        `;
    }

    /**
     * Extract relevant product details for display
     */
    extractProductDetails(product, maxDetails) {
        const priorityFields = [
            'BRAND_DES',
            'MODEL_COD', 
            'USERGENDER_DES',
            'MACRO_SHAPE_AWS',
            'GRANULAR_SHAPE_AWS',
            'RIM_TYPE_DES',
            'CTM_FIRST_FRONT_MATERIAL_DES',
            'COLOR_FAMILY_1_DES',
            'ACT_SKU_PRICE_VAL',
            'MD_SKU_STATUS_COD'
        ];

        const details = [];
        
        // Add priority fields first
        for (const field of priorityFields) {
            const value = product[field];
            if (value && value !== '' && value !== 'null' && details.length < maxDetails) {
                details.push({
                    label: this.formatFieldName(field),
                    value: this.formatFieldValue(field, value)
                });
            }
        }

        // Add other relevant fields if space available
        if (details.length < maxDetails) {
            const otherFields = Object.keys(product).filter(key => 
                !priorityFields.includes(key) && 
                !key.startsWith('Source_') && 
                !key.startsWith('Input_') &&
                !['similarity_score', 'Similarity_Score', 'Similar_SKU', 'filename_root'].includes(key)
            );

            for (const field of otherFields) {
                const value = product[field];
                if (value && value !== '' && value !== 'null' && details.length < maxDetails) {
                    details.push({
                        label: this.formatFieldName(field),
                        value: this.formatFieldValue(field, value)
                    });
                }
            }
        }

        return details;
    }

    /**
     * Format field names for display
     */
    formatFieldName(fieldName) {
        return fieldName
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase())
            .replace(/Des$/, '')
            .replace(/Cod$/, '')
            .replace(/Val$/, '')
            .trim();
    }

    /**
     * Format field values for display
     */
    formatFieldValue(fieldName, value) {
        if (fieldName.includes('PRICE')) {
            const numValue = parseFloat(value);
            if (!isNaN(numValue)) {
                return `€${numValue.toFixed(2)}`;
            }
        }
        
        if (typeof value === 'string' && value.length > 30) {
            return value.substring(0, 30) + '...';
        }
        
        return value;
    }

    /**
     * Get similarity score from product data
     */
    getSimilarityScore(product) {
        return product.similarity_score || product.Similarity_Score || null;
    }

    /**
     * Get SKU from product data
     */
    getSKU(product) {
        return product.SKU_COD || product.Similar_SKU || product.sku || 'Unknown';
    }

    /**
     * Get filename root for image loading
     */
    getFilenameRoot(product) {
        return product.filename_root || product.Similar_filename_root || product.Source_filename_root || null;
    }

    /**
     * Load card image asynchronously
     */
    async loadCardImage(card, product) {
        const filenameRoot = this.getFilenameRoot(product);
        const imageContainer = card.querySelector('.product-image-loading');
        
        if (!filenameRoot || !imageContainer) {
            this.showImagePlaceholder(imageContainer);
            return;
        }

        // Check cache first
        if (this.imageCache.has(filenameRoot)) {
            const imageUrl = this.imageCache.get(filenameRoot);
            if (imageUrl) {
                this.displayImage(imageContainer, imageUrl, filenameRoot);
            } else {
                this.showImagePlaceholder(imageContainer);
            }
            return;
        }

        // Avoid duplicate requests
        if (this.loadingQueue.has(filenameRoot)) {
            return;
        }

        this.loadingQueue.add(filenameRoot);

        try {
            console.log(`🔍 Loading image for: ${filenameRoot}`);
            const response = await fetch(`/api/image/${filenameRoot}`);
            const data = await response.json();
            
            console.log(`📸 Image API response:`, data);
            
            if (data.image_url) {
                this.imageCache.set(filenameRoot, data.image_url);
                this.displayImage(imageContainer, data.image_url, filenameRoot);
            } else {
                console.log(`❌ No image URL for: ${filenameRoot}`);
                this.imageCache.set(filenameRoot, null);
                this.showImagePlaceholder(imageContainer);
            }
        } catch (error) {
            console.error('❌ Error loading image:', error);
            this.imageCache.set(filenameRoot, null);
            this.showImagePlaceholder(imageContainer);
        } finally {
            this.loadingQueue.delete(filenameRoot);
        }
    }

    /**
     * Display loaded image
     */
    displayImage(container, imageUrl, alt) {
        if (container) {
            console.log(`🖼️ Displaying image: ${imageUrl}`);
            const img = document.createElement('img');
            img.src = imageUrl;
            img.alt = alt;
            img.className = 'product-image';
            
            // Add loading and error handlers
            img.onload = () => {
                console.log(`✅ Image loaded successfully: ${imageUrl}`);
                container.innerHTML = '';
                container.appendChild(img);
            };
            
            img.onerror = () => {
                console.error(`❌ Failed to load image: ${imageUrl}`);
                this.showImagePlaceholder(container);
            };
            
            // Start loading
            container.innerHTML = '<div class="product-image-loading"><i class="fas fa-spinner fa-spin"></i><span>Loading...</span></div>';
        }
    }

    /**
     * Show image placeholder
     */
    showImagePlaceholder(container) {
        if (container) {
            container.innerHTML = `
                <div class="product-image-placeholder">
                    <i class="fas fa-image"></i>
                    <span>No image</span>
                </div>
            `;
        }
    }

    /**
     * Attach event listeners to card
     */
    attachCardEvents(card, product) {
        // Card click handler
        card.addEventListener('click', (e) => {
            // Don't trigger on overlay button clicks
            if (!e.target.closest('.product-overlay')) {
                this.handleCardClick(product, card);
            }
        });

        // View details button
        const viewButton = card.querySelector('.btn-view-details');
        if (viewButton) {
            viewButton.addEventListener('click', (e) => {
                e.stopPropagation();
                this.handleViewDetails(product, card);
            });
        }

        // Add hover effects
        card.addEventListener('mouseenter', () => {
            card.classList.add('hovered');
        });

        card.addEventListener('mouseleave', () => {
            card.classList.remove('hovered');
        });
    }

    /**
     * Handle card click
     */
    handleCardClick(product, card) {
        // Dispatch custom event for external handling
        const event = new CustomEvent('productCardClick', {
            detail: { product, card }
        });
        window.dispatchEvent(event);
    }

    /**
     * Handle view details click
     */
    handleViewDetails(product, card) {
        // Dispatch custom event for external handling
        const event = new CustomEvent('productViewDetails', {
            detail: { product, card }
        });
        window.dispatchEvent(event);
    }

    /**
     * Create a source product card (for batch viewer)
     */
    createSourceCard(product, options = {}) {
        const sourceProduct = this.extractSourceProduct(product);
        
        return this.createCard(sourceProduct, {
            ...options,
            size: 'large',
            layout: 'horizontal',
            showSimilarity: false,
            showDetails: true,
            maxDetails: 8
        });
    }

    /**
     * Extract source product information
     */
    extractSourceProduct(result) {
        const sourceProduct = {};
        
        // Extract source/input fields
        Object.keys(result).forEach(key => {
            if (key.startsWith('Source_') || key.startsWith('Input_')) {
                const newKey = key.replace(/^(Source_|Input_)/, '');
                sourceProduct[newKey] = result[key];
            }
        });

        return sourceProduct;
    }

    /**
     * Clear image cache
     */
    clearImageCache() {
        this.imageCache.clear();
    }

    /**
     * Preload images for better performance
     */
    preloadImages(products, maxConcurrent = 5) {
        const filenameRoots = products
            .map(p => this.getFilenameRoot(p))
            .filter(f => f && !this.imageCache.has(f))
            .slice(0, maxConcurrent);

        filenameRoots.forEach(filenameRoot => {
            // Create temporary container for preloading
            const tempContainer = document.createElement('div');
            tempContainer.style.display = 'none';
            document.body.appendChild(tempContainer);
            
            this.loadCardImage({ querySelector: () => tempContainer }, { filename_root: filenameRoot })
                .finally(() => {
                    document.body.removeChild(tempContainer);
                });
        });
    }
} 