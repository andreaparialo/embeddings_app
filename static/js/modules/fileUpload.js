// ================================================
// File Upload Manager Module
// ================================================

export class FileUploadManager {
    constructor() {
        this.dropZones = [];
    }

    init() {
        this.setupFileDropZone('image-drop-zone', 'image-file');
        this.setupFileDropZone('excel-drop-zone', 'excel-file');
    }

    setupFileDropZone(dropZoneId, inputId) {
        const dropZone = document.getElementById(dropZoneId);
        const fileInput = document.getElementById(inputId);
        
        if (!dropZone || !fileInput) return;
        
        this.dropZones.push({ dropZone, fileInput });
        
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });
        
        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => this.highlight(dropZone), false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => this.unhighlight(dropZone), false);
        });
        
        // Handle dropped files
        dropZone.addEventListener('drop', (e) => this.handleDrop(e, dropZone, fileInput), false);
        
        // Handle file input change
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e, dropZone));
        
        // Click on drop zone triggers file input
        dropZone.addEventListener('click', () => fileInput.click());
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlight(dropZone) {
        dropZone.classList.add('drag-over');
    }

    unhighlight(dropZone) {
        dropZone.classList.remove('drag-over');
    }

    handleDrop(e, dropZone, fileInput) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        this.handleFiles(files, dropZone, fileInput);
    }

    handleFileSelect(e, dropZone) {
        const files = e.target.files;
        this.updateDropZoneText(dropZone, files);
    }

    handleFiles(files, dropZone, fileInput) {
        if (files.length === 0) return;
        
        // Validate file type based on input accept attribute
        const accept = fileInput.getAttribute('accept');
        const file = files[0];
        
        if (!this.validateFile(file, accept)) {
            this.showError(`Invalid file type. Expected: ${accept}`);
            return;
        }
        
        // Set files to input
        fileInput.files = files;
        
        // Update UI
        this.updateDropZoneText(dropZone, files);
        
        // Trigger change event
        const event = new Event('change', { bubbles: true });
        fileInput.dispatchEvent(event);
    }

    validateFile(file, accept) {
        if (!accept) return true;
        
        const acceptedTypes = accept.split(',').map(type => type.trim());
        
        for (const acceptedType of acceptedTypes) {
            if (acceptedType.startsWith('.')) {
                // File extension check
                if (file.name.toLowerCase().endsWith(acceptedType.toLowerCase())) {
                    return true;
                }
            } else if (acceptedType.includes('*')) {
                // MIME type with wildcard
                const [mainType] = acceptedType.split('/');
                if (file.type.startsWith(mainType)) {
                    return true;
                }
            } else {
                // Exact MIME type
                if (file.type === acceptedType) {
                    return true;
                }
            }
        }
        
        return false;
    }

    updateDropZoneText(dropZone, files) {
        const textElement = dropZone.querySelector('p:first-of-type');
        if (!textElement) return;
        
        if (files.length > 0) {
            const file = files[0];
            const fileSize = this.formatFileSize(file.size);
            textElement.innerHTML = `<strong>Selected:</strong> ${file.name} (${fileSize})`;
            
            // Add preview for images
            if (file.type.startsWith('image/')) {
                this.showImagePreview(dropZone, file);
            }
        } else {
            textElement.innerHTML = 'Drop file here or click to browse';
        }
    }

    showImagePreview(dropZone, file) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const icon = dropZone.querySelector('.upload-icon');
            if (icon) {
                const preview = document.createElement('img');
                preview.src = e.target.result;
                preview.style.cssText = 'max-width: 150px; max-height: 150px; object-fit: contain;';
                icon.parentNode.replaceChild(preview, icon);
            }
        };
        
        reader.readAsDataURL(file);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showError(message) {
        if (window.visionSearchApp) {
            window.visionSearchApp.showError(message);
        } else {
            alert(message);
        }
    }

    resetDropZone(dropZoneId) {
        const dropZone = document.getElementById(dropZoneId);
        if (!dropZone) return;
        
        // Reset text
        const textElement = dropZone.querySelector('p:first-of-type');
        if (textElement) {
            textElement.innerHTML = 'Drop file here or click to browse';
        }
        
        // Remove preview if exists
        const preview = dropZone.querySelector('img');
        if (preview) {
            const icon = document.createElement('i');
            icon.className = 'fas fa-cloud-upload-alt upload-icon';
            preview.parentNode.replaceChild(icon, preview);
        }
    }
} 