# Vision Search Frontend Architecture

## Overview

The Vision Search frontend has been refactored into a modern, modular architecture that follows best practices for maintainability, scalability, and code organization.

## Directory Structure

```
static/
├── css/
│   └── main.css          # Main stylesheet with organized sections
├── js/
│   ├── app.js           # Main application entry point
│   ├── modules/         # Feature-specific modules
│   │   ├── navigation.js    # Navigation and view management
│   │   ├── loading.js       # Loading state management
│   │   ├── results.js       # Search results display
│   │   ├── filters.js       # Filter controls and management
│   │   ├── fileUpload.js    # Drag-and-drop file uploads
│   │   ├── forms.js         # Form submissions and API calls
│   │   └── batchViewer.js   # Batch results viewer (TODO)
│   └── utils/           # Utility functions
│       └── api.js           # API client for HTTP requests
└── images/
    └── no-image.svg     # Placeholder for missing product images
```

## Architecture Principles

### 1. Modular Design
- Each feature is contained in its own module
- Modules are ES6 classes with clear responsibilities
- Dependencies are explicitly imported

### 2. Separation of Concerns
- **CSS**: All styles in external stylesheet, organized by component
- **HTML**: Clean markup without inline styles or scripts
- **JS**: Business logic separated from presentation

### 3. Component Communication
- Main app instance coordinates between modules
- Custom events for cross-component communication
- Centralized state management through app instance

## Key Modules

### NavigationManager
- Handles view switching and routing
- Manages mobile menu functionality
- Updates page titles and active states
- Supports browser history with hash routing

### LoadingManager
- Centralized loading state management
- Stack-based loading counter for concurrent operations
- Custom loading messages

### ResultsManager
- Renders search results in grid layout
- Handles product card creation
- Manages result counts and pagination
- Click handlers for detailed views (TODO)

### FilterManager
- Dynamically creates filter controls
- Manages filter state collection
- Handles matching column checkboxes
- Column name formatting

### FileUploadManager
- Drag-and-drop file upload zones
- File type validation
- Preview generation for images
- Progress feedback

### FormManager
- Handles all form submissions
- API integration
- Error handling
- Success notifications

### ApiClient
- Centralized HTTP request handling
- Support for JSON, FormData, and Blob responses
- Consistent error handling

## CSS Architecture

The main stylesheet is organized into sections:

1. **CSS Variables & Theme** - Design tokens and theming
2. **Reset & Base Styles** - Normalize browser defaults
3. **Typography** - Font styles and hierarchy
4. **Layout Components** - Major structural elements
5. **Navigation** - Sidebar and header styles
6. **Forms & Controls** - Input and button styles
7. **Cards & Panels** - Content containers
8. **Results Grid** - Product card layouts
9. **Animations** - Transitions and keyframes
10. **Utilities** - Helper classes
11. **Responsive Design** - Mobile adaptations

## Future Enhancements

### Batch Viewer Implementation
The batch viewer module is scaffolded but needs implementation:
- API endpoints for batch result sessions
- Interactive navigation between SKUs
- Result comparison features
- Export functionality

### Additional Features
- Toast notification system
- Advanced filtering UI
- Product comparison mode
- Keyboard shortcuts
- Offline support with Service Workers
- Image lazy loading optimization

## Development Guidelines

### Adding New Features
1. Create a new module in `/js/modules/`
2. Import and initialize in `app.js`
3. Add corresponding CSS in organized sections
4. Update HTML with semantic markup

### Styling Guidelines
- Use CSS variables for consistency
- Follow BEM naming for complex components
- Mobile-first responsive design
- Accessible color contrasts

### JavaScript Guidelines
- Use ES6+ features (classes, modules, arrow functions)
- Async/await for asynchronous operations
- Proper error handling and user feedback
- Document complex logic with comments

## Performance Considerations

- **Code Splitting**: Modules are loaded on demand
- **Image Optimization**: Lazy loading for product images
- **CSS Performance**: Minimal specificity, efficient selectors
- **Caching**: Leverage browser cache for static assets
- **Bundle Size**: Keep dependencies minimal

## Browser Support

- Modern browsers (Chrome, Firefox, Safari, Edge)
- ES6 module support required
- CSS Grid and Flexbox support
- Intersection Observer for lazy loading 