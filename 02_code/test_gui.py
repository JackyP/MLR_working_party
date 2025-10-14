#!/usr/bin/env python3
"""
Test script to run the GUI and take screenshots.
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap

# Add the 02_code directory to the path
sys.path.insert(0, '/home/runner/work/MLR_working_party/MLR_working_party/02_code')

from mlr_gui_app import MLRWorkingPartyGUI


def take_screenshot(window, filename):
    """Take a screenshot of the window."""
    pixmap = window.grab()
    output_path = f'/tmp/{filename}'
    pixmap.save(output_path)
    print(f"Screenshot saved: {output_path}")
    return output_path


def main():
    """Run the GUI and take screenshots."""
    app = QApplication(sys.argv)
    
    # Create the main window
    window = MLRWorkingPartyGUI()
    window.show()
    
    # Take initial screenshot after a short delay
    QTimer.singleShot(500, lambda: take_screenshot(window, 'gui_main_window.png'))
    
    # Open Analyze Data Insights menu after 1 second
    def open_analyze_insights():
        take_screenshot(window, 'gui_before_insights.png')
        window.analyze_data_insights()
        QTimer.singleShot(500, lambda: take_screenshot(window, 'gui_analyze_insights.png'))
        QTimer.singleShot(1000, lambda: app.quit())
    
    QTimer.singleShot(1000, open_analyze_insights)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
