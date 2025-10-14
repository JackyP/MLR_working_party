# Quick Start Guide - MLR GUI Application

Get up and running with the MLR Working Party GUI in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Linux (with X11), macOS, or Windows

## Step 1: Install Dependencies (1 minute)

```bash
# Navigate to the project directory
cd MLR_working_party

# Install Python packages
pip install -r requirements.txt
```

**Note for Linux users:** You may need additional system packages:
```bash
sudo apt-get install -y libegl1 libxkbcommon-x11-0 libxcb-cursor0
```

## Step 2: Generate Sample Data (30 seconds)

```bash
# Generate test data
python3 02_code/generate_sample_data.py
```

This creates `01_data/sample_data.csv` with 500 rows and 14 features.

## Step 3: Launch the Application (10 seconds)

```bash
# Run the GUI
python3 run_gui.py
```

The application window will open automatically.

## Step 4: Load and Analyze Data (2 minutes)

1. **Open Analyze Data Insights**
   - Click: `Manage Experiments` → `Analyze Data Insights`

2. **Load the sample data**
   - Click `Browse...` button
   - Navigate to `01_data/sample_data.csv`
   - Click `Load Dataset`

3. **Explore the tabs**
   - **Data Preview**: See visual distributions of features
   - **Features**: View comprehensive statistics table
   - **Feature Lists**: Create custom feature combinations
   - **Data Insights**: Analyze correlations and dependencies

## Step 5: Try Different Analyses (1 minute)

In the **Data Insights** tab, try each analysis type:

1. **Correlation Matrix**
   - Select from dropdown
   - Click `Generate Insights`
   - View the heatmap

2. **Feature Associations**
   - Select from dropdown
   - Click `Generate Insights`
   - See scatter plots

3. **Pairwise Correlations**
   - Select from dropdown
   - Click `Generate Insights`
   - Review correlation table

4. **Feature Dependencies**
   - Select from dropdown
   - Click `Generate Insights`
   - Check for redundant features

## Your Own Data

To use your own CSV data:

1. Click `Browse...` in the Analyze Data Insights window
2. Select your CSV file
3. Click `Load Dataset`

**Requirements:**
- CSV format with header row
- Numeric or categorical columns
- Any size (tested up to 100,000 rows)

## Troubleshooting

### Problem: "ImportError: No module named PyQt6"
**Solution:** Install dependencies: `pip install PyQt6`

### Problem: "libEGL.so.1: cannot open shared object file"
**Solution:** Install system libraries (Linux):
```bash
sudo apt-get install -y libegl1 libxkbcommon-x11-0 libxcb-cursor0
```

### Problem: Application doesn't start
**Solution:** Check Python version: `python3 --version` (need 3.8+)

### Problem: Can't see visualizations
**Solution:** Try installing matplotlib: `pip install matplotlib seaborn`

## Running Tests

To verify everything works:

```bash
# Generate screenshots (requires xvfb on Linux)
cd 02_code
xvfb-run -a python3 test_gui_simple.py

# Run comprehensive demo
xvfb-run -a python3 demo_gui.py
```

Screenshots will be saved in the `screenshots/` directory.

## What's Next?

- **Read the docs**: Check out [README.md](README.md) for full documentation
- **Visual guide**: See [VISUAL_GUIDE.md](VISUAL_GUIDE.md) for annotated screenshots
- **Detailed docs**: Read [02_code/README_GUI.md](02_code/README_GUI.md) for technical details

## Key Features at a Glance

✅ **Data Preview**
- Visual feature distributions
- Automatic type detection
- Summary statistics

✅ **Features Analysis**
- Comprehensive table view
- Missing value analysis
- Detailed statistics

✅ **Feature Lists**
- Create custom lists
- Multi-select interface
- Retrain integration

✅ **Data Insights**
- Correlation analysis
- Feature associations
- Statistical tests
- Dependency detection

## Need Help?

- Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for implementation details
- Review screenshots in `screenshots/` directory
- Run the demo: `python3 02_code/demo_gui.py`

## Command Reference

```bash
# Launch GUI
python3 run_gui.py

# Generate sample data
python3 02_code/generate_sample_data.py

# Run demo
cd 02_code && xvfb-run -a python3 demo_gui.py

# Run tests
cd 02_code && xvfb-run -a python3 test_gui_simple.py

# Install dependencies
pip install -r requirements.txt
```

---

**Total setup time:** < 5 minutes
**First analysis:** < 2 minutes
**Difficulty:** Easy

Ready to go! 🚀
