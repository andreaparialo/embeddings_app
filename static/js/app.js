// ================================================
// Vision Search - Main Application
// ================================================

import { NavigationManager } from './modules/navigation.js?v=5';
import { FormManager } from './modules/forms.js?v=5';
import { ResultsManager } from './modules/results.js?v=5';
import { FilterManager } from './modules/filters.js?v=5';
import { LoadingManager } from './modules/loading.js?v=5';
import { BatchViewerManager } from './modules/batchViewer.js?v=5';
import { FileUploadManager } from './modules/fileUpload.js?v=5';
import { SidebarComponent } from './modules/sidebar.js?v=5';
import { ProductCardComponent } from './modules/productCard.js?v=5';

class VisionSearchApp {
    constructor() {
        this.filterOptions = window.filterOptions || {};
        this.initializationStatus = window.initializationStatus || { initialized: false };
        
        // Initialize components
        this.sidebar = new SidebarComponent();
        this.productCard = new ProductCardComponent();
        
        // Initialize managers
        this.navigation = new NavigationManager();
        this.loading = new LoadingManager();
        this.results = new ResultsManager();
        this.filters = new FilterManager(this.filterOptions);
        this.fileUpload = new FileUploadManager();
        this.forms = new FormManager(this);
        this.batchViewer = new BatchViewerManager();
    }

    init() {
        console.log('🚀 Initializing Vision Search App');
        
        try {
            // Initialize components first
            this.sidebar.init();
            console.log('✅ Sidebar initialized');
            
            // Initialize all modules with error handling
            try {
                this.navigation.init();
                console.log('✅ Navigation initialized');
            } catch (e) {
                console.error('❌ Navigation initialization failed:', e);
            }
            
            try {
                this.filters.init();
                console.log('✅ Filters initialized');
            } catch (e) {
                console.error('❌ Filters initialization failed:', e);
            }
            
            try {
                this.fileUpload.init();
                console.log('✅ File upload initialized');
            } catch (e) {
                console.error('❌ File upload initialization failed:', e);
            }
            
            try {
                this.forms.init();
                console.log('✅ Forms initialized');
            } catch (e) {
                console.error('❌ Forms initialization failed:', e);
            }
            
            try {
                console.log('🔥 Initializing batch viewer...');
                console.log('🔥 BatchViewer instance:', this.batchViewer);
                this.batchViewer.init();
                console.log('✅ Batch viewer initialized');
            } catch (e) {
                console.error('❌ Batch viewer initialization failed:', e);
            }
            
            // Set initial status
            this.updateSystemStatus();
            
            // Attach global event listeners
            this.attachGlobalEvents();
            
            console.log('✅ Vision Search App initialized');
        } catch (error) {
            console.error('❌ Critical error during app initialization:', error);
            // Try to at least show the navigation
            if (this.navigation) {
                this.navigation.showView('image-search');
            }
        }
    }

    attachGlobalEvents() {
        // Listen for sidebar events
        window.addEventListener('sidebarToggle', (event) => {
            console.log('Sidebar toggled:', event.detail.collapsed);
            // Optional: save state or trigger other actions
        });

        window.addEventListener('responsiveChange', (event) => {
            console.log('Responsive mode changed:', event.detail.isMobile);
            // Handle responsive changes if needed
        });

        // Listen for product card events
        window.addEventListener('productCardClick', (event) => {
            const { product, card } = event.detail;
            console.log('Product card clicked:', product);
            // Handle product card clicks - could show details modal, navigate, etc.
        });

        window.addEventListener('productViewDetails', (event) => {
            const { product, card } = event.detail;
            console.log('Product view details:', product);
            // Handle view details - could show detailed modal, etc.
            this.showProductDetails(product);
        });

        // Listen for view changes to update components
        window.addEventListener('viewChanged', (event) => {
            const { viewId } = event.detail;
            console.log('View changed to:', viewId);
            
            // Update sidebar state if needed
            if (viewId === 'batch-viewer') {
                // Maybe expand sidebar for batch viewer
                this.sidebar.expand();
            }
        });
    }

    showProductDetails(product) {
        // Create a simple product details modal
        const modal = document.createElement('div');
        modal.className = 'product-details-modal';
        modal.innerHTML = `
            <div class="modal-backdrop"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Product Details</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="product-details-grid">
                        ${Object.entries(product).map(([key, value]) => 
                            value ? `<div class="detail-item">
                                <span class="detail-key">${key.replace(/_/g, ' ')}</span>
                                <span class="detail-value">${value}</span>
                            </div>` : ''
                        ).join('')}
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Close modal handlers
        const closeModal = () => {
            modal.remove();
        };

        modal.querySelector('.modal-close').addEventListener('click', closeModal);
        modal.querySelector('.modal-backdrop').addEventListener('click', closeModal);

        // Close on escape key
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                closeModal();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
    }

    updateSystemStatus() {
        const statusBadge = document.getElementById('status-badge');
        const statusText = document.getElementById('status-text');
        
        if (this.initializationStatus.initialized) {
            statusBadge.classList.remove('error');
            statusText.textContent = 'System Ready';
        } else {
            statusBadge.classList.add('error');
            statusText.textContent = this.initializationStatus.message || 'System Not Ready';
        }
    }

    showLoading() {
        this.loading.show();
    }

    hideLoading() {
        this.loading.hide();
    }

    showResults(results, containerId, countId) {
        this.results.showResults(results, containerId, countId);
    }

    showError(message) {
        // Use the batch viewer's notification system
        if (this.batchViewer.showNotification) {
            this.batchViewer.showNotification(message, 'error');
        } else {
            alert(`Error: ${message}`);
        }
    }

    showSuccess(message) {
        // Use the batch viewer's notification system
        if (this.batchViewer.showNotification) {
            this.batchViewer.showNotification(message, 'success');
        } else {
            alert(`Success: ${message}`);
        }
    }

    // New utility methods for component access
    getSidebar() {
        return this.sidebar;
    }

    getProductCard() {
        return this.productCard;
    }

    // Method to toggle sidebar programmatically
    toggleSidebar() {
        this.sidebar.toggleCollapse();
    }

    // Method to check if sidebar is collapsed
    isSidebarCollapsed() {
        return this.sidebar.isCollapsedState();
    }
}

// Expose global functions immediately for testing
console.log('📦 Exposing global fallback functions...');

window.toggleSidebar = () => {
    console.log('🔄 toggleSidebar called via fallback');
    if (window.visionSearchApp?.getSidebar) {
        window.visionSearchApp.getSidebar().toggleCollapse();
    } else {
        console.error('VisionSearchApp not ready yet');
    }
};

window.collapseSidebar = () => {
    console.log('📉 collapseSidebar called via fallback');
    if (window.visionSearchApp?.getSidebar) {
        window.visionSearchApp.getSidebar().collapse();
    } else {
        console.error('VisionSearchApp not ready yet');
    }
};

window.expandSidebar = () => {
    console.log('📈 expandSidebar called via fallback');
    if (window.visionSearchApp?.getSidebar) {
        window.visionSearchApp.getSidebar().expand();
    } else {
        console.error('VisionSearchApp not ready yet');
    }
};

console.log('✅ Global fallback functions exposed');

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 DOM is ready, initializing VisionSearchApp...');
    
    try {
        window.visionSearchApp = new VisionSearchApp();
        console.log('📱 VisionSearchApp created successfully');
        
        window.visionSearchApp.init();
        console.log('✅ VisionSearchApp initialized successfully');
        
        // Update global functions to use the app instance
        window.toggleSidebar = () => {
            console.log('🔄 toggleSidebar called');
            if (window.visionSearchApp?.getSidebar) {
                window.visionSearchApp.getSidebar().toggleCollapse();
            } else {
                console.error('VisionSearchApp not available');
            }
        };

        window.collapseSidebar = () => {
            console.log('📉 collapseSidebar called');
            if (window.visionSearchApp?.getSidebar) {
                window.visionSearchApp.getSidebar().collapse();
            } else {
                console.error('VisionSearchApp not available');
            }
        };

        window.expandSidebar = () => {
            console.log('📈 expandSidebar called');
            if (window.visionSearchApp?.getSidebar) {
                window.visionSearchApp.getSidebar().expand();
            } else {
                console.error('VisionSearchApp not available');
            }
        };
        
        // Expose debugging functions globally
        window.testButtonUpdate = () => {
            const instance = window.visionSearchApp?.forms;
            if (instance) {
                instance.updateSubmitButtonText();
            } else {
                console.error('FormManager instance not found');
            }
        };
        
        // Also expose a simple manual toggle for testing
        window.toggleViewerMode = () => {
            const checkbox = document.getElementById('viewer-mode');
            if (checkbox) {
                checkbox.checked = !checkbox.checked;
                checkbox.dispatchEvent(new Event('change'));
                console.log('Viewer mode toggled to:', checkbox.checked);
            } else {
                console.error('Viewer mode checkbox not found');
            }
        };
        
        // Debug button function
        window.debugUploadButton = () => {
            console.log('🔍 Debugging upload button...');
            const btn = document.getElementById('upload-excel-btn');
            if (btn) {
                console.log('✅ Button found:', btn);
                console.log('📄 Button HTML:', btn.outerHTML);
                console.log('👂 Event listeners:', getEventListeners ? getEventListeners(btn) : 'getEventListeners not available');
                
                // Try clicking it programmatically
                console.log('🖱️ Triggering click event...');
                btn.click();
            } else {
                console.error('❌ Button not found');
                console.log('🔍 All buttons:', document.querySelectorAll('button'));
            }
        };
        
        console.log('🌐 Global functions updated successfully');
        
    } catch (error) {
        console.error('❌ Error during VisionSearchApp initialization:', error);
        console.error('Stack trace:', error.stack);
        
        // Provide error feedback to user
        document.body.insertAdjacentHTML('afterbegin', `
            <div style="background: red; color: white; padding: 10px; position: fixed; top: 0; left: 0; right: 0; z-index: 9999;">
                ❌ JavaScript Error: ${error.message} - Check console for details
            </div>
        `);
    }
}); 