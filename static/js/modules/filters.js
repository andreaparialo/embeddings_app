// ================================================
// Filter Manager Module
// ================================================

export class FilterManager {
    constructor(filterOptions) {
        this.filterOptions = filterOptions || {};
        this.mainFilters = [
            'BRAND_DES', 
            'BRAND_CLUSTER', 
            'USERGENDER_DES', 
            'COLOR_FAMILY_1_DES', 
            'COLOR', 
            'SHAPE_SEMI_GROUPED'
        ];
        
        this.mainColumns = [
            'BRAND_DES', 
            'BRAND_CLUSTER', 
            'USERGENDER_DES', 
            'COLOR_FAMILY_1_DES', 
            'PRODUCT_TYPE_COD',
            'COLOR', 
            'CTM_FIRST_TEMPLE_MATERIAL_DES', 
            'SHAPE_SEMI_GROUPED', 
            'BRIDGE_LENGTH', 
            'LENS_BASE', 
            'LENSHEIGHTVAL'
        ];
    }

    init() {
        this.createFilterControls('image-filters');
        this.createFilterControls('filter-controls');
        this.createMatchingColumnsCheckboxes();
    }

    createFilterControls(containerId, prefix = '') {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        
        // Sort filters to put main filters first
        const sortedFilters = Object.entries(this.filterOptions)
            .filter(([_, options]) => options.length <= 500)
            .sort(([a], [b]) => {
                const aIndex = this.mainFilters.indexOf(a);
                const bIndex = this.mainFilters.indexOf(b);
                if (aIndex !== -1 && bIndex !== -1) return aIndex - bIndex;
                if (aIndex !== -1) return -1;
                if (bIndex !== -1) return 1;
                return 0;
            });
        
        sortedFilters.forEach(([column, options]) => {
            const filterGroup = this.createFilterGroup(column, options, prefix);
            container.appendChild(filterGroup);
        });
    }

    createFilterGroup(column, options, prefix = '') {
        const div = document.createElement('div');
        div.className = 'form-group';
        
        const label = document.createElement('label');
        label.className = 'form-label';
        label.textContent = this.formatColumnName(column);
        label.htmlFor = `${prefix}${column}`;
        
        const select = document.createElement('select');
        select.className = 'form-select';
        select.name = prefix + column;
        select.id = `${prefix}${column}`;
        
        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = '-- All --';
        select.appendChild(defaultOption);
        
        // Add options
        options.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option;
            opt.textContent = option;
            select.appendChild(opt);
        });
        
        div.appendChild(label);
        div.appendChild(select);
        
        return div;
    }

    createMatchingColumnsCheckboxes() {
        const container = document.getElementById('excel-matching-columns');
        if (!container) return;
        
        container.innerHTML = '';
        
        // Create checkboxes for main columns
        this.mainColumns.forEach(column => {
            if (this.filterOptions[column]) {
                const checkbox = this.createColumnCheckbox(column);
                container.appendChild(checkbox);
            }
        });
        
        // Add other columns in collapsible section
        const otherColumns = Object.keys(this.filterOptions)
            .filter(col => !this.mainColumns.includes(col) && 
                          this.filterOptions[col].length < 500);
        
        if (otherColumns.length > 0) {
            const details = this.createOtherColumnsSection(otherColumns);
            container.appendChild(details);
        }
    }

    createColumnCheckbox(column) {
        const colDiv = document.createElement('div');
        colDiv.className = 'col-md-4 col-sm-6 mb-2';
        
        const checkDiv = document.createElement('div');
        checkDiv.className = 'form-check';
        
        const checkbox = document.createElement('input');
        checkbox.className = 'form-check-input';
        checkbox.type = 'checkbox';
        checkbox.id = `match-${column}`;
        checkbox.name = 'matching_columns';
        checkbox.value = column;
        
        const label = document.createElement('label');
        label.className = 'form-check-label';
        label.htmlFor = `match-${column}`;
        label.textContent = this.formatColumnName(column);
        
        checkDiv.appendChild(checkbox);
        checkDiv.appendChild(label);
        colDiv.appendChild(checkDiv);
        
        return colDiv;
    }

    createOtherColumnsSection(columns) {
        const detailsDiv = document.createElement('div');
        detailsDiv.className = 'col-12 mt-2';
        
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'Other Columns';
        
        const otherContainer = document.createElement('div');
        otherContainer.className = 'row';
        
        columns.forEach(column => {
            const checkbox = this.createColumnCheckbox(column);
            otherContainer.appendChild(checkbox);
        });
        
        details.appendChild(summary);
        details.appendChild(otherContainer);
        detailsDiv.appendChild(details);
        
        return detailsDiv;
    }

    collectFilters(containerElement) {
        const filters = {};
        
        if (!containerElement) return filters;
        
        const inputs = containerElement.querySelectorAll('input, select');
        
        inputs.forEach(input => {
            if (input.name && input.value && input.value.trim() !== '') {
                filters[input.name] = input.value.trim();
            }
        });
        
        return filters;
    }

    collectStatusCodes(formId) {
        const statusCodes = [];
        const form = document.getElementById(formId);
        
        if (form) {
            form.querySelectorAll('input[name="status_filter"]:checked').forEach(checkbox => {
                statusCodes.push(checkbox.value);
            });
        }
        
        return statusCodes;
    }

    collectMatchingColumns() {
        const selectedColumns = [];
        const checkboxes = document.querySelectorAll('input[name="matching_columns"]:checked');
        
        checkboxes.forEach(checkbox => {
            selectedColumns.push(checkbox.value);
        });
        
        return selectedColumns;
    }

    formatColumnName(column) {
        return column.replace(/_/g, ' ')
                    .split(' ')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                    .join(' ');
    }

    resetFilters(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Reset all selects
        container.querySelectorAll('select').forEach(select => {
            select.selectedIndex = 0;
        });
        
        // Uncheck all checkboxes
        container.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.checked = false;
        });
    }
} 