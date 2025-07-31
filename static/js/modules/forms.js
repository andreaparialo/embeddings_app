// ================================================
// Form Manager Module
// ================================================

import { ApiClient } from '../utils/api.js?v=5';

export class FormManager {
    constructor(app) {
        this.app = app;
        this.api = new ApiClient();
    }

    init() {
        this.attachFormHandlers();
        // Use a more robust approach to handle viewer mode
        this.initViewerModeHandler();
    }

    attachFormHandlers() {
        // Image search form
        const imageForm = document.getElementById('image-search-form');
        if (imageForm) {
            imageForm.addEventListener('submit', (e) => this.handleImageSearch(e));
        }

        // SKU search form
        const skuForm = document.getElementById('sku-search-form');
        if (skuForm) {
            skuForm.addEventListener('submit', (e) => this.handleSkuSearch(e));
        }

        // Excel batch search form
        const excelForm = document.getElementById('excel-search-form');
        if (excelForm) {
            excelForm.addEventListener('submit', (e) => this.handleExcelSearch(e));
            
            // Viewer mode handling is now done in initViewerModeHandler()
        }

        // Filter search form
        const filterForm = document.getElementById('filter-search-form');
        if (filterForm) {
            filterForm.addEventListener('submit', (e) => this.handleFilterSearch(e));
        }

        // Settings forms
        const checkpointForm = document.getElementById('checkpoint-form');
        if (checkpointForm) {
            checkpointForm.addEventListener('submit', (e) => this.handleCheckpointChange(e));
        }

        const indexForm = document.getElementById('index-form');
        if (indexForm) {
            indexForm.addEventListener('submit', (e) => this.handleIndexChange(e));
        }
    }

    initViewerModeHandler() {
        // Use event delegation to handle viewer mode changes
        document.addEventListener('change', (e) => {
            if (e.target && e.target.id === 'viewer-mode') {
                this.updateSubmitButtonText();
            }
        });
        
        // Initial button text update
        setTimeout(() => {
            this.updateSubmitButtonText();
        }, 100);
    }

    async handleImageSearch(e) {
        e.preventDefault();
        
        // Check if dual engine is enabled
        const dualEngineEnabled = document.getElementById('dual-engine-image')?.checked || false;
        const searchType = dualEngineEnabled ? 'dual-index' : 'standard';
        
        this.app.showLoading(`Searching for similar products using ${searchType} search...`);
        
        const formData = new FormData(e.target);
        const filters = this.app.filters.collectFilters(document.getElementById('image-filters'));
        formData.append('filters', JSON.stringify(filters));
        
        // Add dual engine parameters if enabled
        let endpoint = '/search/image';
        if (dualEngineEnabled) {
            endpoint = '/search/image-dual';
            const mainWeight = parseFloat(document.getElementById('main-weight')?.value || 0.7);
            const measurementWeight = parseFloat(document.getElementById('measurement-weight')?.value || 0.3);
            
            formData.append('main_weight', mainWeight);
            formData.append('measurement_weight', measurementWeight);
            
            console.log(`🔍 Using dual-index search with weights: Visual=${mainWeight}, Technical=${measurementWeight}`);
        }
        
        try {
            const result = await this.api.post(endpoint, formData);
            
            if (result.error) {
                this.app.showError(result.error);
                this.app.hideLoading();
            } else {
                this.app.showResults(
                    result.results, 
                    'image-results-container', 
                    'image-results-count'
                );
            }
        } catch (error) {
            this.app.showError(error.message);
            this.app.hideLoading();
        }
    }

    async handleSkuSearch(e) {
        e.preventDefault();
        
        // Always use enhanced SKU search with image similarity
        this.app.showLoading('Performing SKU search with image similarity...');
        
        const formData = new FormData();
        formData.append('sku', document.getElementById('sku-input').value);
        
        // Collect matching columns
        const selectedColumns = [];
        document.querySelectorAll('#sku-matching-columns input[type="checkbox"]:checked').forEach(checkbox => {
            selectedColumns.push(checkbox.value);
        });
        formData.append('matching_columns', JSON.stringify(selectedColumns));
        
        // Check if dual engine is enabled for SKU search
        const dualEngineEnabled = document.getElementById('sku-dual-engine')?.checked || false;
        
        // Add other options
        formData.append('dual_engine', dualEngineEnabled);
        formData.append('exclude_same_model', document.getElementById('sku-exclude-same-model')?.checked || false);
        formData.append('group_unisex', document.getElementById('sku-group-unisex')?.checked || false);
        formData.append('top_k', document.getElementById('sku-top-k')?.value || 20);
        formData.append('allowed_status_codes', JSON.stringify(['IL'])); // Default to IL status
        
        // Add weight parameters if dual engine is enabled
        if (dualEngineEnabled) {
            const mainWeight = parseFloat(document.getElementById('sku-main-weight')?.value || 0.7);
            const measurementWeight = parseFloat(document.getElementById('sku-measurement-weight')?.value || 0.3);
            
            formData.append('main_weight', mainWeight);
            formData.append('measurement_weight', measurementWeight);
            
            console.log(`🔍 SKU dual-index search with weights: Visual=${mainWeight}, Technical=${measurementWeight}`);
        }
        
        try {
            const result = await this.api.post('/search/sku', formData);
            
            if (result.error) {
                this.app.showError(result.error);
                this.app.hideLoading();
            } else {
                // Show results with source SKU info
                if (result.source_sku) {
                    console.log('Source SKU:', result.source_sku);
                    console.log('Pre-filters applied:', result.prefilters_applied);
                    console.log('Post-filters applied:', result.postfilters_applied);
                    if (result.dual_engine) {
                        console.log('Dual engine weights:', { 
                            main: result.main_weight || 0.7, 
                            measurement: result.measurement_weight || 0.3 
                        });
                    }
                }
                
                this.app.showResults(
                    result.results || [], 
                    'sku-results-container', 
                    'sku-results-count'
                );
            }
        } catch (error) {
            this.app.showError(error.message);
            this.app.hideLoading();
        }
    }

    updateSubmitButtonText() {
        // Try multiple ways to find the elements
        const viewerModeCheckbox = document.getElementById('viewer-mode') || 
                                 document.querySelector('input[name="viewer_mode"]') ||
                                 document.querySelector('#viewer-mode');
        
        const submitButtonText = document.getElementById('submit-button-text') ||
                               document.querySelector('#submit-button-text') ||
                               document.querySelector('button[type="submit"] span');
        
        const viewerMode = viewerModeCheckbox ? viewerModeCheckbox.checked : false;
        
        console.log('updateSubmitButtonText called:', { 
            viewerMode, 
            viewerModeCheckbox: !!viewerModeCheckbox,
            submitButtonText: !!submitButtonText,
            submitButtonTextContent: submitButtonText ? submitButtonText.textContent : 'not found'
        });
        
        if (submitButtonText) {
            if (viewerMode) {
                submitButtonText.textContent = 'Process & View Results';
            } else {
                submitButtonText.textContent = 'Process & Export Results';
            }
            console.log('Button text updated to:', submitButtonText.textContent);
        } else {
            console.warn('Submit button text element not found!');
            // Fallback: try to find any submit button and update its text
            const submitButton = document.querySelector('#excel-search-form button[type="submit"]');
            if (submitButton) {
                const span = submitButton.querySelector('span') || submitButton;
                if (viewerMode) {
                    span.innerHTML = '<i class="fas fa-upload"></i> <span>Process & View Results</span>';
                } else {
                    span.innerHTML = '<i class="fas fa-upload"></i> <span>Process & Export Results</span>';
                }
                console.log('Fallback: Updated button via innerHTML');
            }
        }
    }
    
    // Expose this function globally for debugging
    static testButtonUpdate() {
        const instance = window.visionSearchApp?.forms;
        if (instance) {
            instance.updateSubmitButtonText();
        } else {
            console.error('FormManager instance not found');
        }
    }

    async handleExcelSearch(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        
        // Collect matching columns
        const selectedColumns = this.app.filters.collectMatchingColumns();
        
        if (selectedColumns.length === 0) {
            this.app.showError('Please select at least one column to match.');
            return;
        }
        
        formData.append('matching_columns', JSON.stringify(selectedColumns));
        formData.append('exclude_same_model', document.getElementById('exclude-same-model').checked);
        
        // Collect status codes
        const selectedStatusCodes = [];
        document.querySelectorAll('input[name="allowed_status_codes"]:checked').forEach(checkbox => {
            selectedStatusCodes.push(checkbox.value);
        });
        formData.append('allowed_status_codes', JSON.stringify(selectedStatusCodes));
        
        formData.append('group_unisex', document.getElementById('group-unisex').checked);
        
        // Check if dual engine is enabled
        const dualEngineEnabled = document.getElementById('dual-engine').checked;
        formData.append('dual_engine', dualEngineEnabled);
        
        // Add dual engine parameters if enabled
        if (dualEngineEnabled) {
            // Get weight values
            const mainWeight = parseFloat(document.getElementById('batch-main-weight')?.value || 0.7);
            const measurementWeight = parseFloat(document.getElementById('batch-measurement-weight')?.value || 0.3);
            formData.append('main_weight', mainWeight);
            formData.append('measurement_weight', measurementWeight);
            
            // Get search mode (global or filtered)
            const searchMode = document.querySelector('input[name="search_mode"]:checked')?.value || 'global';
            formData.append('search_mode', searchMode);
            
            console.log(`🔍 Batch dual-index search: mode=${searchMode}, weights: Visual=${mainWeight}, Technical=${measurementWeight}`);
        }
        
        const filterOnlyMode = document.getElementById('filter-only-mode').checked;
        const viewerMode = document.getElementById('viewer-mode').checked;
        
        console.log('Batch search settings:', {
            filterOnlyMode,
            viewerMode,
            willUseViewer: viewerMode && !filterOnlyMode
        });
        
        // Set return_session flag for viewer mode
        if (!filterOnlyMode) {
            formData.append('return_session', viewerMode);
        }
        
        const endpoint = filterOnlyMode ? '/search/batch-filter-only' : '/search/batch-enhanced';
        
        this.app.showLoading('Processing batch search...');
        
        try {
            if (viewerMode && !filterOnlyMode) {
                // For viewer mode, expect JSON response with session ID
                const result = await this.api.post(endpoint, formData);
                
                if (result.error) {
                    this.app.showError(result.error);
                } else if (result.session_id) {
                    this.app.showSuccess(`Batch processing completed! Found ${result.total_results} results for ${result.total_skus} SKUs.`);
                    
                    // Load results in batch viewer
                    await this.app.batchViewer.handleBatchSearchComplete(result.session_id);
                } else {
                    this.app.showError('Unexpected response from server.');
                }
            } else {
                // For download mode, expect blob response
                const blob = await this.api.postBlob(endpoint, formData);
                
                // Download the file
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'batch_search_results.xlsx';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.app.showSuccess('Results exported successfully! Check your downloads.');
            }
            
            this.app.hideLoading();
        } catch (error) {
            this.app.showError(error.message);
            this.app.hideLoading();
        }
    }

    async handleFilterSearch(e) {
        e.preventDefault();
        
        this.app.showLoading('Searching with filters...');
        
        const filters = this.app.filters.collectFilters(document.getElementById('filter-controls'));
        const statusCodes = this.app.filters.collectStatusCodes('filter-search-form');
        
        if (statusCodes.length > 0) {
            filters['MD_SKU_STATUS_COD'] = statusCodes;
        }
        
        const formData = new FormData();
        formData.append('filters', JSON.stringify(filters));
        
        try {
            const result = await this.api.post('/search/filters', formData);
            
            if (result.error) {
                this.app.showError(result.error);
                this.app.hideLoading();
            } else {
                this.app.showResults(
                    result.results, 
                    'filter-results-container', 
                    'filter-results-count'
                );
            }
        } catch (error) {
            this.app.showError(error.message);
            this.app.hideLoading();
        }
    }

    async handleCheckpointChange(e) {
        e.preventDefault();
        
        this.app.showLoading('Switching checkpoint...');
        
        const formData = new FormData(e.target);
        
        try {
            const result = await this.api.post('/api/change-checkpoint', formData);
            
            if (result.error) {
                this.app.showError(result.error);
            } else {
                this.app.showSuccess(result.message);
                setTimeout(() => location.reload(), 1500);
            }
        } catch (error) {
            this.app.showError(error.message);
        } finally {
            this.app.hideLoading();
        }
    }

    async handleIndexChange(e) {
        e.preventDefault();
        
        this.app.showLoading('Switching index...');
        
        const formData = new FormData(e.target);
        
        try {
            const result = await this.api.post('/api/change-index', formData);
            
            if (result.error) {
                this.app.showError(result.error);
            } else {
                this.app.showSuccess(result.message);
                setTimeout(() => location.reload(), 1500);
            }
        } catch (error) {
            this.app.showError(error.message);
        } finally {
            this.app.hideLoading();
        }
    }
} 