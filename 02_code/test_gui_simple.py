#!/usr/bin/env python3
"""
Simple test script for the GUI - no dialogs.
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


def main():
    """Test the GUI."""
    app = QApplication(sys.argv)
    
    # Create main window
    window = MLRWorkingPartyGUI()
    window.show()
    app.processEvents()
    
    # Screenshot 1: Main window
    QTimer.singleShot(100, lambda: take_screenshot(window, '01_main_window.png'))
    
    # Screenshot 2: Open Analyze Data Insights
    def step2():
        window.analyze_data_insights()
        app.processEvents()
        QTimer.singleShot(100, lambda: take_screenshot(window, '02_insights_empty.png'))
        QTimer.singleShot(200, step3)
    
    # Screenshot 3: Set data path and load (without dialog)
    def step3():
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                # Bypass the dialog - just set the path directly
                data_path = '/home/runner/work/MLR_working_party/MLR_working_party/01_data/sample_data.csv'
                insights_widget.path_input.setText(data_path)
                app.processEvents()
                QTimer.singleShot(100, lambda: take_screenshot(window, '03_data_path_set.png'))
                QTimer.singleShot(200, step4)
    
    # Screenshot 4: After loading data (suppress dialog)
    def step4():
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                # Directly load without showing dialogs
                import pandas as pd
                data_path = insights_widget.path_input.text()
                insights_widget.data = pd.read_csv(data_path)
                insights_widget.current_dataset_path = data_path
                insights_widget.status_label.setText(
                    f"Dataset loaded: {os.path.basename(data_path)} "
                    f"({insights_widget.data.shape[0]} rows, {insights_widget.data.shape[1]} columns)"
                )
                insights_widget.status_label.setStyleSheet("padding: 5px; color: green;")
                insights_widget.update_data_preview()
                insights_widget.update_features_table()
                insights_widget.update_available_features()
                app.processEvents()
                QTimer.singleShot(500, lambda: take_screenshot(window, '04_data_loaded.png'))
                QTimer.singleShot(600, step5)
    
    # Screenshot 5: Features tab
    def step5():
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                insights_widget.tabs.setCurrentIndex(1)
                app.processEvents()
                QTimer.singleShot(100, lambda: take_screenshot(window, '05_features_tab.png'))
                QTimer.singleShot(200, step6)
    
    # Screenshot 6: Feature Lists tab
    def step6():
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                insights_widget.tabs.setCurrentIndex(2)
                app.processEvents()
                QTimer.singleShot(100, lambda: take_screenshot(window, '06_feature_lists.png'))
                QTimer.singleShot(200, step7)
    
    # Screenshot 7: Data Insights tab
    def step7():
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                insights_widget.tabs.setCurrentIndex(3)
                app.processEvents()
                QTimer.singleShot(100, lambda: take_screenshot(window, '07_insights_tab.png'))
                QTimer.singleShot(200, step8)
    
    # Screenshot 8: Generate correlation matrix
    def step8():
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                insights_widget.analysis_type_combo.setCurrentIndex(0)
                insights_widget.generate_data_insights()
                app.processEvents()
                QTimer.singleShot(1000, lambda: take_screenshot(window, '08_correlation_matrix.png'))
                QTimer.singleShot(1100, step9)
    
    # Screenshot 9: Feature associations
    def step9():
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                insights_widget.analysis_type_combo.setCurrentIndex(1)
                insights_widget.generate_data_insights()
                app.processEvents()
                QTimer.singleShot(1000, lambda: take_screenshot(window, '09_feature_associations.png'))
                QTimer.singleShot(1100, lambda: app.quit())
    
    # Start the sequence
    QTimer.singleShot(100, step2)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
