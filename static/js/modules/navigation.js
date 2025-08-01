// ================================================
// Navigation Manager Module
// ================================================

export class NavigationManager {
    constructor() {
        this.views = document.querySelectorAll('.view-container');
        this.navItems = document.querySelectorAll('.nav-item');
        this.pageTitle = document.getElementById('page-title');
        this.sidebar = document.getElementById('sidebar');
        
        this.viewTitles = {
            'image-search': 'Image Search',
            'sku-search': 'SKU Search',
            'batch-search': 'Batch Search',
            'filter-search': 'Filter Search',
            'batch-viewer': 'Batch Viewer',
            'settings': 'Settings'
        };
    }

    init() {
        this.attachEventListeners();
        this.initMobileMenu();
        
        // Show initial view based on URL hash or default
        const initialView = window.location.hash.slice(1) || 'image-search';
        this.showView(initialView);
    }

    attachEventListeners() {
        this.navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const viewId = item.dataset.view;
                this.showView(viewId);
                
                // Update URL hash
                window.location.hash = viewId;
                
                // Close mobile menu if open
                if (window.innerWidth <= 768) {
                    this.closeSidebar();
                }
            });
        });

        // Handle browser back/forward
        window.addEventListener('hashchange', () => {
            const viewId = window.location.hash.slice(1) || 'image-search';
            this.showView(viewId);
        });
    }

    showView(viewId) {
        console.log('Showing view:', viewId);
        
        // First, clear all results to prevent overlap
        this.clearAllResults();
        
        // Hide all views by removing active class
        this.views.forEach(view => {
            view.classList.remove('active');
        });
        
        // Show selected view
        const targetView = document.getElementById(viewId + '-view');
        if (targetView) {
            // Add active class immediately
            targetView.classList.add('active');
            console.log('Added active class to:', viewId + '-view');
        } else {
            console.error('View not found:', viewId + '-view');
        }
        
        // Update navigation active state
        this.navItems.forEach(item => {
            item.classList.remove('active');
            if (item.dataset.view === viewId) {
                item.classList.add('active');
            }
        });
        
        // Update page title
        if (this.pageTitle) {
            this.pageTitle.textContent = this.viewTitles[viewId] || 'Vision Search';
        }
        
        // Emit custom event
        window.dispatchEvent(new CustomEvent('viewChanged', { 
            detail: { viewId } 
        }));
    }

    clearAllResults() {
        // Hide all results containers to prevent them from appearing on other views
        const allResultsContainers = [
            'image-results-container',
            'sku-results-container',
            'batch-results-container',
            'filter-results-container'
        ];
        
        allResultsContainers.forEach(containerId => {
            const container = document.getElementById(containerId);
            if (container) {
                container.style.display = 'none';
                
                // Also clear the grid content
                const grid = document.getElementById(containerId.replace('-container', '-grid'));
                if (grid) {
                    grid.innerHTML = '';
                }
                
                // Clear count badge
                const countBadge = document.getElementById(containerId.replace('-container', '-count'));
                if (countBadge) {
                    countBadge.textContent = '';
                }
            }
        });
    }

    initMobileMenu() {
        // Create mobile menu toggle button
        const mobileMenuToggle = document.createElement('button');
        mobileMenuToggle.className = 'mobile-menu-toggle';
        mobileMenuToggle.innerHTML = '<i class="fas fa-bars"></i>';
        mobileMenuToggle.style.display = 'none';
        
        // Insert before page title
        const header = document.querySelector('.header');
        header.insertBefore(mobileMenuToggle, header.firstChild);
        
        // Show/hide toggle based on screen size
        const checkMobile = () => {
            if (window.innerWidth <= 768) {
                mobileMenuToggle.style.display = 'block';
            } else {
                mobileMenuToggle.style.display = 'none';
                this.sidebar.classList.remove('open');
            }
        };
        
        window.addEventListener('resize', checkMobile);
        checkMobile();
        
        // Toggle menu on click
        mobileMenuToggle.addEventListener('click', () => {
            this.toggleSidebar();
        });
        
        // Close menu on outside click
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768 && 
                !this.sidebar.contains(e.target) && 
                !mobileMenuToggle.contains(e.target) && 
                this.sidebar.classList.contains('open')) {
                this.closeSidebar();
            }
        });
    }

    toggleSidebar() {
        this.sidebar.classList.toggle('open');
    }

    closeSidebar() {
        this.sidebar.classList.remove('open');
    }

    openSidebar() {
        this.sidebar.classList.add('open');
    }
} 