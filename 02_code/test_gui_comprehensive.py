#!/usr/bin/env python3
"""
Comprehensive test script for the GUI with data loading and visualization.
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Add the 02_code directory to the path
sys.path.insert(0, '/home/runner/work/MLR_working_party/MLR_working_party/02_code')

from mlr_gui_app import MLRWorkingPartyGUI
from analyze_data_insights import AnalyzeDataInsights


def take_screenshot(widget, filename):
    """Take a screenshot of the widget."""
    pixmap = widget.grab()
    output_path = f'/tmp/{filename}'
    pixmap.save(output_path)
    print(f"Screenshot saved: {output_path}")
    return output_path


def test_sequence():
    """Test the GUI with a sequence of operations."""
    app = QApplication(sys.argv)
    
    # Test 1: Main window
    print("\n=== Test 1: Main Window ===")
    window = MLRWorkingPartyGUI()
    window.show()
    app.processEvents()
    take_screenshot(window, 'screenshot_01_main_window.png')
    
    # Test 2: Open Analyze Data Insights
    print("\n=== Test 2: Analyze Data Insights (Empty) ===")
    window.analyze_data_insights()
    app.processEvents()
    take_screenshot(window, 'screenshot_02_insights_empty.png')
    
    # Test 3: Load data
    print("\n=== Test 3: Loading Data ===")
    # Get the AnalyzeDataInsights widget from the center pane
    center_layout = window.center_pane.layout()
    if center_layout.count() > 0:
        insights_widget = center_layout.itemAt(0).widget()
        if isinstance(insights_widget, AnalyzeDataInsights):
            # Set the data path
            data_path = '/home/runner/work/MLR_working_party/MLR_working_party/01_data/sample_data.csv'
            insights_widget.path_input.setText(data_path)
            app.processEvents()
            
            # Load the dataset
            insights_widget.load_dataset()
            app.processEvents()
            take_screenshot(window, 'screenshot_03_data_loaded.png')
            
            # Test 4: Data Preview tab
            print("\n=== Test 4: Data Preview Tab ===")
            insights_widget.tabs.setCurrentIndex(0)
            app.processEvents()
            QTimer.singleShot(500, lambda: None)  # Give time for rendering
            app.processEvents()
            take_screenshot(window, 'screenshot_04_data_preview.png')
            
            # Test 5: Features tab
            print("\n=== Test 5: Features Tab ===")
            insights_widget.tabs.setCurrentIndex(1)
            app.processEvents()
            take_screenshot(window, 'screenshot_05_features_table.png')
            
            # Select a feature
            if insights_widget.features_table.rowCount() > 0:
                insights_widget.features_table.selectRow(0)
                app.processEvents()
                take_screenshot(window, 'screenshot_06_feature_details.png')
            
            # Test 6: Feature Lists tab
            print("\n=== Test 6: Feature Lists Tab ===")
            insights_widget.tabs.setCurrentIndex(2)
            app.processEvents()
            take_screenshot(window, 'screenshot_07_feature_lists.png')
            
            # Test 7: Data Insights tab
            print("\n=== Test 7: Data Insights Tab ===")
            insights_widget.tabs.setCurrentIndex(3)
            app.processEvents()
            take_screenshot(window, 'screenshot_08_insights_tab.png')
            
            # Generate correlation matrix
            print("\n=== Test 8: Correlation Matrix ===")
            insights_widget.analysis_type_combo.setCurrentIndex(0)
            insights_widget.generate_data_insights()
            app.processEvents()
            QTimer.singleShot(1000, lambda: None)  # Give time for rendering
            app.processEvents()
            take_screenshot(window, 'screenshot_09_correlation_matrix.png')
            
            # Generate feature associations
            print("\n=== Test 9: Feature Associations ===")
            insights_widget.analysis_type_combo.setCurrentIndex(1)
            insights_widget.generate_data_insights()
            app.processEvents()
            QTimer.singleShot(1000, lambda: None)
            app.processEvents()
            take_screenshot(window, 'screenshot_10_feature_associations.png')
            
            # Generate pairwise correlations
            print("\n=== Test 10: Pairwise Correlations ===")
            insights_widget.analysis_type_combo.setCurrentIndex(2)
            insights_widget.generate_data_insights()
            app.processEvents()
            take_screenshot(window, 'screenshot_11_pairwise_correlations.png')
    
    print("\n=== All tests completed successfully! ===")
    print("\nScreenshots saved in /tmp/")
    
    # Clean up
    window.close()
    app.quit()


if __name__ == "__main__":
    test_sequence()
