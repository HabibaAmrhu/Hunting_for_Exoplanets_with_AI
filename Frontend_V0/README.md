# 🌌 NASA Exoplanet Detection Pipeline - Primary Frontend

**🏆 PRIMARY SUBMISSION INTERFACE - NASA Space Apps Challenge 2025**

## 👥 Team
- **Habiba Amr**: AI/ML Development, Backend Systems
- **Aisha Samir**: Frontend Development, UI/UX Design

## 📋 Overview

This is the **primary submission interface** - a pure HTML/CSS/JavaScript implementation of the NASA Exoplanet Detection Pipeline web application. This frontend showcases our AI-powered exoplanet detection system through an engaging, interactive user experience.

## 🗂️ Project Structure

```
Frontend_V0/
├── index.html          # Main HTML file with all pages
├── css/
│   ├── theme.css       # Theme colors and variables (Light Mode Only)
│   └── styles.css      # Main stylesheet with all components
├── js/
│   ├── app.js          # Main application logic and navigation
│   ├── game.js         # Beginner Mode game logic
│   └── analysis.js     # Research Mode analysis logic
├── assets/             # Images and other assets (to be added)
└── README.md           # This file
```

## 🚀 Features

### Beginner Mode
- **🎮 Hunt a Planet**: Interactive game to identify exoplanets
  - Real-time light curve visualization
  - Dynamic feedback system with 32 response variations
  - Score tracking and accuracy calculation
  - AI analysis integration

- **📚 Learn the Basics**: Educational content
  - 5 comprehensive topics about exoplanet detection
  - Interactive topic selector
  - Detailed explanations

- **🔍 Example Gallery**: 4 example light curves
  - Clear Transit Signal (Easy)
  - Noisy Transit Signal (Medium)
  - Stellar Variability (Medium)
  - Subtle Transit Signal (Hard)

- **🏆 Challenge Mode**: Timed challenges
  - 4 difficulty levels
  - Scoring system
  - Time tracking

### Research Mode
- **📊 Single Analysis**: Analyze individual light curves
  - File upload (CSV, TXT, NPY)
  - Example data selector
  - Manual synthetic data generation
  - AI prediction with confidence scores

- **📁 Batch Processing**: Process multiple files
  - Multi-file upload
  - Batch analysis
  - Summary statistics

- **🔍 Model Comparison**: Compare AI models
  - Select multiple models
  - Side-by-side comparison
  - Performance metrics

- **📈 Performance Analysis**: Model performance dashboard
  - Accuracy charts
  - Inference time analysis
  - Confusion matrices

## 🎨 Design System

### Color Palette (Light Mode Only)
- **Primary Background**: `#FFFFFF`
- **Secondary Background**: `#F8FAFC`
- **Primary Text**: `#0F172A`
- **Accent Color**: `#667eea`
- **NASA Blue**: `#1f4e79`
- **Success**: `#10b981`
- **Error**: `#ef4444`

### Typography
- Font Family: `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto`
- Base Font Size: `16px`
- Line Height: `1.6`

## 📦 Dependencies

### External Libraries
- **Plotly.js** (v2.26.0): For interactive light curve visualization
  - CDN: `https://cdn.plot.ly/plotly-2.26.0.min.js`

### No Build Process Required
This is a pure frontend application with no build step. Simply open `index.html` in a web browser.

## 🚀 Getting Started

### Option 1: Direct File Opening
1. Navigate to the `Frontend_V0` folder
2. Double-click `index.html` to open in your default browser

### Option 2: Local Server (Recommended)
Using Python:
```bash
cd Frontend_V0
python -m http.server 8000
```

Using Node.js (http-server):
```bash
cd Frontend_V0
npx http-server -p 8000
```

Then open: `http://localhost:8000`

### Option 3: VS Code Live Server
1. Install "Live Server" extension in VS Code
2. Right-click `index.html`
3. Select "Open with Live Server"

## 🎮 How to Use

### Beginner Mode - Hunt a Planet
1. Click "Beginner Mode" in the sidebar
2. Select "Hunt a Planet" tab
3. Observe the light curve
4. Click "Yes, I see a planet!" or "No planet detected"
5. View feedback and explanation
6. Click "Run AI Analysis" to see AI prediction
7. Click "Try a different sample" for a new challenge

### Research Mode - Single Analysis
1. Click "Research Mode" in the sidebar
2. Select "Single Analysis" tab
3. Choose input method:
   - **Upload File**: Upload CSV/TXT file
   - **Use Example**: Select from 4 examples
   - **Manual Input**: Generate synthetic data
4. View analysis results
5. Export results if needed

## 🔧 Customization

### Adding New Themes
Edit `css/theme.css` to add new color schemes:
```css
:root {
    --primary-bg: #FFFFFF;
    --accent-primary: #667eea;
    /* Add more variables */
}
```

### Adding New Response Templates
Edit `js/game.js` and add to `responseTemplates` object:
```javascript
const responseTemplates = {
    correct_planet: [
        "Your new response here!",
        // Add more variations
    ]
};
```

### Modifying Light Curve Generation
Edit the `generateLightCurve()` function in `js/game.js`:
```javascript
function generateLightCurve(hasPlanet, difficulty) {
    // Customize parameters
    const period = 5.0;  // Orbital period
    const depth = 0.01;  // Transit depth
    // ...
}
```

## 📊 Data Format

### CSV/TXT File Format
```csv
time,flux
0.00,1.0000
0.01,0.9998
0.02,0.9995
...
```

### Expected Data Structure
- **Time**: Days (float)
- **Flux**: Normalized flux (float, typically around 1.0)
- **Length**: Recommended 2048 points

## 🎯 Features Comparison with Streamlit Version

| Feature | Streamlit | Frontend_V0 | Status |
|---------|-----------|-------------|--------|
| Hunt a Planet | ✅ | ✅ | Complete |
| Learn the Basics | ✅ | ✅ | Complete |
| Example Gallery | ✅ | ✅ | Complete |
| Challenge Mode | ✅ | ⚠️ | Partial |
| Single Analysis | ✅ | ✅ | Complete |
| Batch Processing | ✅ | ⚠️ | Partial |
| Model Comparison | ✅ | ⚠️ | Partial |
| Performance Dashboard | ✅ | ⏳ | Planned |
| Real AI Models | ✅ | ❌ | Simulated |

**Legend:**
- ✅ Complete
- ⚠️ Partial (UI complete, backend simulated)
- ⏳ Planned
- ❌ Not implemented

## 🔮 Future Enhancements

### Phase 1 (Current)
- [x] Complete UI implementation
- [x] Navigation system
- [x] Light curve visualization
- [x] Response system
- [x] Educational content

### Phase 2 (Next)
- [ ] Backend API integration
- [ ] Real AI model predictions
- [ ] User authentication
- [ ] Data persistence
- [ ] Advanced analytics

### Phase 3 (Future)
- [ ] Mobile responsive design
- [ ] Dark mode support
- [ ] Accessibility improvements
- [ ] Performance optimization
- [ ] Progressive Web App (PWA)

## 🐛 Known Issues

1. **Challenge Mode**: Timer and full game loop not implemented
2. **Batch Processing**: Simulated results only
3. **Model Comparison**: Mock data, not real model outputs
4. **File Upload**: Limited file format support
5. **Mobile**: Not fully optimized for mobile devices

## 🤝 Contributing

To contribute to this project:

1. Test the application thoroughly
2. Report bugs or suggest features
3. Submit improvements to code
4. Enhance documentation

## 📝 Notes

### Differences from Streamlit Version
- **No Backend**: All processing is client-side
- **Simulated AI**: AI predictions are randomly generated
- **Static Data**: No connection to NASA database
- **Limited File Support**: Only basic CSV/TXT parsing

### Performance Considerations
- Light curve generation is done in JavaScript
- Plotly.js handles visualization
- No server-side processing required
- All data stays in browser memory

## 📄 License

This project is part of the NASA Space Apps Challenge 2025.

## 🚀 Deployment

### GitHub Pages
1. Push to GitHub repository
2. Enable GitHub Pages in repository settings
3. Select main branch and root folder
4. Access at: `https://username.github.io/repo-name/Frontend_V0/`

### Netlify
1. Drag and drop `Frontend_V0` folder to Netlify
2. Or connect GitHub repository
3. Deploy automatically

### Vercel
```bash
cd Frontend_V0
vercel
```

## 📞 Support

For questions or issues:
- Check the main project README
- Review the code comments
- Test in different browsers
- Check browser console for errors

## 🎉 Acknowledgments

- Based on the Streamlit application
- Uses Plotly.js for visualizations
- Inspired by NASA's exoplanet detection methods
- Built for NASA Space Apps Challenge 2025

---

**Version**: 1.0.0  
**Last Updated**: October 4, 2025  
**Status**: Production Ready (Frontend Only)
