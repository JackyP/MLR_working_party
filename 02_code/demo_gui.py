#!/usr/bin/env python3
"""
Comprehensive demonstration script for the MLR GUI.
This script showcases all features of the Analyze Data Insights module.
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

sys.path.insert(0, '/home/runner/work/MLR_working_party/MLR_working_party/02_code')

from mlr_gui_app import MLRWorkingPartyGUI
from analyze_data_insights import AnalyzeDataInsights


def take_screenshot(widget, filename, description=""):
    """Take a screenshot of the widget."""
    pixmap = widget.grab()
    output_path = f'/home/runner/work/MLR_working_party/MLR_working_party/screenshots/{filename}'
    pixmap.save(output_path)
    if description:
        print(f"✓ {description}")
    print(f"  Screenshot: {filename}")


def demo():
    """Run comprehensive demonstration."""
    app = QApplication(sys.argv)
    
    print("\n" + "="*70)
    print("MLR Working Party GUI - Comprehensive Demonstration")
    print("="*70)
    
    # Create main window
    window = MLRWorkingPartyGUI()
    window.show()
    app.processEvents()
    
    print("\n1. Main Application Window")
    print("-" * 70)
    QTimer.singleShot(100, lambda: take_screenshot(
        window, 'demo_01_main_window.png',
        "Main window with 3-pane layout and menu bar"
    ))
    
    # Open Analyze Data Insights
    def step2():
        print("\n2. Analyze Data Insights - Initial View")
        print("-" * 70)
        window.analyze_data_insights()
        app.processEvents()
        QTimer.singleShot(100, lambda: take_screenshot(
            window, 'demo_02_insights_initial.png',
            "Data Insights module opened (empty state)"
        ))
        QTimer.singleShot(200, step3)
    
    # Load data
    def step3():
        print("\n3. Loading Dataset")
        print("-" * 70)
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                import pandas as pd
                data_path = '/home/runner/work/MLR_working_party/MLR_working_party/01_data/sample_data.csv'
                
                # Load data directly
                insights_widget.data = pd.read_csv(data_path)
                insights_widget.current_dataset_path = data_path
                insights_widget.path_input.setText(data_path)
                insights_widget.status_label.setText(
                    f"Dataset loaded: {os.path.basename(data_path)} "
                    f"({insights_widget.data.shape[0]} rows, {insights_widget.data.shape[1]} columns)"
                )
                insights_widget.status_label.setStyleSheet("padding: 5px; color: green;")
                insights_widget.update_data_preview()
                insights_widget.update_features_table()
                insights_widget.update_available_features()
                app.processEvents()
                
                print(f"  Loaded: {insights_widget.data.shape[0]} rows, {insights_widget.data.shape[1]} columns")
                QTimer.singleShot(500, lambda: take_screenshot(
                    window, 'demo_03_data_preview.png',
                    "Data Preview tab with visual feature representations"
                ))
                QTimer.singleShot(600, step4)
    
    # Features tab
    def step4():
        print("\n4. Features Analysis")
        print("-" * 70)
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                insights_widget.tabs.setCurrentIndex(1)
                app.processEvents()
                
                # Select first feature
                if insights_widget.features_table.rowCount() > 0:
                    insights_widget.features_table.selectRow(0)
                    app.processEvents()
                
                print(f"  Features table with {insights_widget.features_table.rowCount()} features")
                QTimer.singleShot(100, lambda: take_screenshot(
                    window, 'demo_04_features_table.png',
                    "Features table with statistics and detail view"
                ))
                QTimer.singleShot(200, step5)
    
    # Feature Lists tab
    def step5():
        print("\n5. Feature Lists Management")
        print("-" * 70)
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                insights_widget.tabs.setCurrentIndex(2)
                app.processEvents()
                
                # Select some features
                for i in range(min(5, insights_widget.available_features_list.count())):
                    insights_widget.available_features_list.item(i).setSelected(True)
                app.processEvents()
                
                print("  Feature lists with multi-select interface")
                QTimer.singleShot(100, lambda: take_screenshot(
                    window, 'demo_05_feature_lists.png',
                    "Feature Lists tab with selection interface"
                ))
                QTimer.singleShot(200, step6)
    
    # Data Insights - Correlation Matrix
    def step6():
        print("\n6. Correlation Matrix Analysis")
        print("-" * 70)
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                insights_widget.tabs.setCurrentIndex(3)
                insights_widget.analysis_type_combo.setCurrentIndex(0)
                insights_widget.generate_data_insights()
                app.processEvents()
                
                print("  Correlation matrix heatmap generated")
                QTimer.singleShot(1000, lambda: take_screenshot(
                    window, 'demo_06_correlation_matrix.png',
                    "Correlation matrix heatmap visualization"
                ))
                QTimer.singleShot(1100, step7)
    
    # Data Insights - Feature Associations
    def step7():
        print("\n7. Feature Associations")
        print("-" * 70)
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                insights_widget.analysis_type_combo.setCurrentIndex(1)
                insights_widget.generate_data_insights()
                app.processEvents()
                
                print("  Feature association scatter plots generated")
                QTimer.singleShot(1000, lambda: take_screenshot(
                    window, 'demo_07_feature_associations.png',
                    "Top feature associations with scatter plots"
                ))
                QTimer.singleShot(1100, step8)
    
    # Data Insights - Pairwise Correlations
    def step8():
        print("\n8. Pairwise Correlations")
        print("-" * 70)
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                insights_widget.analysis_type_combo.setCurrentIndex(2)
                insights_widget.generate_data_insights()
                app.processEvents()
                
                print("  Pairwise correlation table with Pearson and Spearman")
                QTimer.singleShot(500, lambda: take_screenshot(
                    window, 'demo_08_pairwise_correlations.png',
                    "Detailed pairwise correlation analysis"
                ))
                QTimer.singleShot(600, step9)
    
    # Data Insights - Feature Dependencies
    def step9():
        print("\n9. Feature Dependencies")
        print("-" * 70)
        center_layout = window.center_pane.layout()
        if center_layout.count() > 0:
            insights_widget = center_layout.itemAt(0).widget()
            if isinstance(insights_widget, AnalyzeDataInsights):
                insights_widget.analysis_type_combo.setCurrentIndex(3)
                insights_widget.generate_data_insights()
                app.processEvents()
                
                print("  Feature dependency analysis completed")
                QTimer.singleShot(1000, lambda: take_screenshot(
                    window, 'demo_09_feature_dependencies.png',
                    "Feature dependencies and redundancy detection"
                ))
                QTimer.singleShot(1100, finish)
    
    def finish():
        print("\n" + "="*70)
        print("✓ Demonstration Complete!")
        print("="*70)
        print("\nAll screenshots saved to: screenshots/")
        print("\nFeatures demonstrated:")
        print("  ✓ 3-pane GUI layout")
        print("  ✓ Menu bar with File, Manage Experiments, About")
        print("  ✓ Data Preview with visual representations")
        print("  ✓ Features table with statistics")
        print("  ✓ Feature Lists management")
        print("  ✓ Correlation Matrix analysis")
        print("  ✓ Feature Associations")
        print("  ✓ Pairwise Correlations")
        print("  ✓ Feature Dependencies")
        print("\nTo run the GUI interactively:")
        print("  python3 run_gui.py")
        print("="*70 + "\n")
        
        app.quit()
    
    # Start the sequence
    QTimer.singleShot(100, step2)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    demo()
