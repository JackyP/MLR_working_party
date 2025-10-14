# MLR Working Party GUI Application

A comprehensive PyQt6-based GUI application for managing machine learning experiments and analyzing data insights.

## Features

### Main Interface
- **3-Pane Layout**: 
  - Left Pane: Navigation and project tree
  - Top Pane: Summary and key metrics
  - Center Pane: Main content area for detailed views

### Menu Bar
1. **File Menu**
   - New Project (Ctrl+N)
   - Open Project (Ctrl+O)
   - Save (Ctrl+S)
   - Exit (Ctrl+Q)

2. **Manage Experiments Menu**
   - Experiment Setup
   - **Analyze Data Insights** (Fully Implemented)
   - Blueprint Repository
   - Model Leaderboard
   - Experiment Insights
   - Compare Models
   - Add/Retrain Models
   - Edit (Composable) Blueprints

3. **About Menu**
   - About
   - Help (F1)

## Analyze Data Insights (Fully Implemented)

The "Analyze Data Insights" functionality provides comprehensive data analysis tools organized into four tiles:

### 1. Data Preview Tile
- Visual representation of features in your dataset
- Displays frequent values and distributions
- Histograms for numeric features
- Bar charts for categorical features
- Adjustable sample size for preview
- Summary statistics for each feature

### 2. Features Tile
- Displays all features in a table format with:
  - Feature name
  - Data type
  - Missing value percentage
  - Number of unique values
  - Statistical measures (Mean, Std, Min, Max)
- Select features to view detailed statistics
- Interactive feature selection for deeper insights

### 3. Feature Lists Tile
- Create custom feature lists from available features
- Manage existing feature lists
- Delete unwanted feature lists
- Retrain models with different feature combinations
- Multi-select interface for easy feature selection

### 4. Data Insights Tile
- **Correlation Matrix**: Heatmap visualization of feature correlations
- **Feature Associations**: Scatter plots of top correlated feature pairs
- **Pairwise Correlations**: Detailed table with Pearson and Spearman correlations
- **Feature Dependencies**: Identify highly dependent features that may be redundant

## Installation

### Requirements
```bash
pip install PyQt6 pandas numpy matplotlib seaborn scipy
```

### System Dependencies (for Linux)
```bash
sudo apt-get install -y xvfb libegl1 libxkbcommon-x11-0 libxcb-cursor0
```

## Usage

### Running the Application
```bash
cd 02_code
python3 mlr_gui_app.py
```

### Loading Data
1. Click "Manage Experiments" → "Analyze Data Insights"
2. Click "Browse..." to select a CSV file or enter the path directly
3. Click "Load Dataset" to load the data
4. Navigate through the tabs to explore different analysis views

### Sample Data
A sample dataset is included for testing:
```bash
# Generate sample data
python3 generate_sample_data.py

# The sample data will be saved to: 01_data/sample_data.csv
```

## Files Structure

```
02_code/
├── mlr_gui_app.py              # Main GUI application
├── analyze_data_insights.py     # Analyze Data Insights module
├── generate_sample_data.py      # Sample data generator
├── test_gui_simple.py          # GUI test script
└── utils/                       # Existing utility modules
    ├── config.py
    ├── data_engineering.py
    └── ...

01_data/
└── sample_data.csv             # Generated sample dataset

screenshots/                     # GUI screenshots
├── 01_main_window.png
├── 02_insights_empty.png
├── 03_data_path_set.png
├── 04_data_loaded.png
├── 05_features_tab.png
├── 06_feature_lists.png
├── 07_insights_tab.png
├── 08_correlation_matrix.png
└── 09_feature_associations.png
```

## Testing

Run the automated test suite:
```bash
cd 02_code
xvfb-run -a python3 test_gui_simple.py
```

This will generate screenshots in `/tmp/` showing the application in action.

## Key Features of Implementation

### Data Preview
- Automatic detection of numeric vs categorical features
- Visual distribution plots for each feature
- Configurable sample size
- Summary statistics

### Features Analysis
- Comprehensive feature table with key metrics
- Interactive selection for detailed view
- Support for both numeric and categorical features
- Missing value analysis

### Feature Lists Management
- Create custom feature lists
- Store multiple feature lists
- Easy selection and management
- Integration with model retraining workflow

### Data Insights Visualization
- Multiple correlation analysis methods
- Interactive visualizations using matplotlib
- Statistical significance testing
- Dependency detection for feature engineering

## Integration with Existing Code

The GUI integrates with the existing MLR Working Party codebase:
- Uses the same data structure from `utils/config.py`
- Compatible with existing data processing pipelines
- Can load data processed by `GRU_framework_NJC.py`
- Feature lists can be used to retrain models

## Future Enhancements

The following menu items are placeholders for future implementation:
- Experiment Setup
- Blueprint Repository
- Model Leaderboard
- Experiment Insights
- Compare Models
- Add/Retrain Models
- Edit (Composable) Blueprints

These can be implemented following the same pattern as the Analyze Data Insights module.

## Screenshots

### Main Window
![Main Window](screenshots/01_main_window.png)

### Analyze Data Insights - Empty
![Insights Empty](screenshots/02_insights_empty.png)

### Data Loaded
![Data Loaded](screenshots/04_data_loaded.png)

### Features Tab
![Features Tab](screenshots/05_features_tab.png)

### Feature Lists
![Feature Lists](screenshots/06_feature_lists.png)

### Correlation Matrix
![Correlation Matrix](screenshots/08_correlation_matrix.png)

### Feature Associations
![Feature Associations](screenshots/09_feature_associations.png)

## License

Part of the MLR Working Party project.

## Contributors

Created for the MLR Working Party to provide a comprehensive GUI for experiment management and data analysis.
