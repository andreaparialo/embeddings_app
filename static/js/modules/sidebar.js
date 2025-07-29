// ================================================
// Sidebar Component Module
// ================================================

export class SidebarComponent {
    constructor() {
        this.sidebar = null;
        this.mainContent = null;
        this.isCollapsed = false;
        this.isMobile = false;
        this.initialized = false;
    }

    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.doInit());
        } else {
            this.doInit();
        }
    }

    doInit() {
        this.sidebar = document.getElementById('sidebar');
        this.mainContent = document.querySelector('.main-content');
        
        if (!this.sidebar || !this.mainContent) {
            console.warn('Sidebar: Required DOM elements not found');
            return;
        }

        this.createToggleButton();
        this.attachEventListeners();
        this.checkMobile();
        this.loadState();
        this.initialized = true;
        
        console.log('✅ Sidebar component initialized');
    }

    createToggleButton() {
        // Only create desktop toggle button (mobile toggle is handled by navigation.js)
        if (this.sidebar.querySelector('.sidebar-toggle')) {
            return; // Already exists
        }

        const toggleButton = document.createElement('button');
        toggleButton.className = 'sidebar-toggle';
        toggleButton.innerHTML = '<i class="fas fa-angle-double-left"></i>';
        toggleButton.title = 'Toggle Sidebar (Ctrl+B)';
        
        // Insert at the top of sidebar logo
        const logo = this.sidebar.querySelector('.logo');
        if (logo) {
            logo.style.position = 'relative';
            logo.appendChild(toggleButton);
        }
    }

    attachEventListeners() {
        // Desktop toggle
        const toggleButton = this.sidebar.querySelector('.sidebar-toggle');
        if (toggleButton) {
            toggleButton.addEventListener('click', () => {
                this.toggleCollapse();
            });
        }

        // Don't create mobile toggle - that's handled by navigation.js
        // Just listen for existing mobile toggle if it exists
        const existingMobileToggle = document.querySelector('.mobile-menu-toggle');
        if (existingMobileToggle) {
            // Override the existing click handler to use our mobile methods
            existingMobileToggle.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleMobile();
            });
        }

        // Close mobile menu on outside click
        document.addEventListener('click', (e) => {
            if (this.isMobile && 
                this.sidebar.classList.contains('mobile-open') &&
                !this.sidebar.contains(e.target) && 
                !e.target.closest('.mobile-menu-toggle')) {
                this.closeMobile();
            }
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            this.checkMobile();
        });

        // Handle keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'b') {
                e.preventDefault();
                this.toggleCollapse();
            }
        });
    }

    toggleCollapse() {
        if (!this.initialized || this.isMobile) return;
        
        this.isCollapsed = !this.isCollapsed;
        
        if (this.isCollapsed) {
            this.sidebar.classList.add('collapsed');
            this.mainContent.classList.add('sidebar-collapsed');
            this.updateToggleIcon('fa-angle-double-right');
        } else {
            this.sidebar.classList.remove('collapsed');
            this.mainContent.classList.remove('sidebar-collapsed');
            this.updateToggleIcon('fa-angle-double-left');
        }
        
        this.saveState();
        this.dispatchEvent('sidebarToggle', { collapsed: this.isCollapsed });
    }

    toggleMobile() {
        if (!this.initialized) return;
        
        // Use the same class name as the original navigation for compatibility
        if (this.sidebar.classList.contains('open')) {
            this.closeMobile();
        } else {
            this.openMobile();
        }
    }

    openMobile() {
        this.sidebar.classList.add('open');
        this.sidebar.classList.add('mobile-open');
        document.body.style.overflow = 'hidden';
    }

    closeMobile() {
        this.sidebar.classList.remove('open');
        this.sidebar.classList.remove('mobile-open');
        document.body.style.overflow = '';
    }

    checkMobile() {
        if (!this.initialized) return;
        
        const wasMobile = this.isMobile;
        this.isMobile = window.innerWidth <= 768;
        
        const mobileToggle = document.querySelector('.mobile-menu-toggle');
        const desktopToggle = this.sidebar.querySelector('.sidebar-toggle');
        
        if (this.isMobile) {
            if (mobileToggle) mobileToggle.style.display = 'block';
            if (desktopToggle) desktopToggle.style.display = 'none';
            
            // Reset desktop collapse state on mobile
            this.sidebar.classList.remove('collapsed');
            this.mainContent.classList.remove('sidebar-collapsed');
        } else {
            if (mobileToggle) mobileToggle.style.display = 'none';
            if (desktopToggle) desktopToggle.style.display = 'flex';
            
            // Close mobile menu
            this.closeMobile();
            
            // Restore desktop collapse state
            if (this.isCollapsed) {
                this.sidebar.classList.add('collapsed');
                this.mainContent.classList.add('sidebar-collapsed');
            }
        }
        
        if (wasMobile !== this.isMobile) {
            this.dispatchEvent('responsiveChange', { isMobile: this.isMobile });
        }
    }

    updateToggleIcon(iconClass) {
        const toggleButton = this.sidebar?.querySelector('.sidebar-toggle i');
        if (toggleButton) {
            toggleButton.className = `fas ${iconClass}`;
        }
    }

    saveState() {
        try {
            localStorage.setItem('sidebarCollapsed', this.isCollapsed);
        } catch (e) {
            // Ignore localStorage errors
        }
    }

    loadState() {
        try {
            const savedState = localStorage.getItem('sidebarCollapsed');
            if (savedState !== null) {
                this.isCollapsed = savedState === 'true';
                if (this.isCollapsed && !this.isMobile) {
                    this.sidebar.classList.add('collapsed');
                    this.mainContent.classList.add('sidebar-collapsed');
                    this.updateToggleIcon('fa-angle-double-right');
                }
            }
        } catch (e) {
            // Ignore localStorage errors
        }
    }

    dispatchEvent(eventName, detail) {
        window.dispatchEvent(new CustomEvent(eventName, { detail }));
    }

    // Public API
    collapse() {
        if (!this.isCollapsed) {
            this.toggleCollapse();
        }
    }

    expand() {
        if (this.isCollapsed) {
            this.toggleCollapse();
        }
    }

    isCollapsedState() {
        return this.isCollapsed;
    }

    isMobileState() {
        return this.isMobile;
    }

    isInitialized() {
        return this.initialized;
    }
} 