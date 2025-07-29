// ================================================
// Loading Manager Module
// ================================================

export class LoadingManager {
    constructor() {
        this.loadingContainer = document.getElementById('loading');
        this.loadingStack = 0;
    }

    show(message = 'Processing your request...') {
        this.loadingStack++;
        
        if (this.loadingStack === 1) {
            const messageElement = this.loadingContainer.querySelector('p');
            if (messageElement) {
                messageElement.textContent = message;
            }
            
            this.loadingContainer.classList.add('active');
            document.body.style.cursor = 'wait';
        }
    }

    hide() {
        this.loadingStack = Math.max(0, this.loadingStack - 1);
        
        if (this.loadingStack === 0) {
            this.loadingContainer.classList.remove('active');
            document.body.style.cursor = '';
        }
    }

    forceHide() {
        this.loadingStack = 0;
        this.hide();
    }

    isLoading() {
        return this.loadingStack > 0;
    }
} 