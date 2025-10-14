#!/usr/bin/env python3
"""
MLR Working Party GUI Application
A PyQt6-based GUI for managing machine learning experiments and analyzing data insights.
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMenuBar, QMenu, QTextEdit, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

# Import the data insights module
from analyze_data_insights import AnalyzeDataInsights


class MLRWorkingPartyGUI(QMainWindow):
    """Main GUI application for MLR Working Party."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MLR Working Party - Experiment Manager")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize the central widget and layout
        self.init_ui()
        self.create_menu_bar()
        
    def init_ui(self):
        """Initialize the user interface with 3-pane layout."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create the main splitter for 3-pane interface
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # LEFT PANE - Navigation/Project tree
        self.left_pane = QWidget()
        left_layout = QVBoxLayout(self.left_pane)
        left_label = QLabel("Navigation Panel")
        left_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        self.left_content = QTextEdit()
        self.left_content.setPlaceholderText("Project navigation and experiment list will appear here...")
        left_layout.addWidget(left_label)
        left_layout.addWidget(self.left_content)
        
        # RIGHT SIDE - Contains top and center panes
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create vertical splitter for top and center panes
        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # TOP PANE - Summary/Header information
        self.top_pane = QWidget()
        top_layout = QVBoxLayout(self.top_pane)
        top_label = QLabel("Summary Panel")
        top_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        self.top_content = QTextEdit()
        self.top_content.setPlaceholderText("Experiment summary and key metrics will appear here...")
        self.top_content.setMaximumHeight(200)
        top_layout.addWidget(top_label)
        top_layout.addWidget(self.top_content)
        
        # CENTER PANE - Main content area
        self.center_pane = QWidget()
        center_layout = QVBoxLayout(self.center_pane)
        center_label = QLabel("Main Content Area")
        center_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        self.center_content = QTextEdit()
        self.center_content.setPlaceholderText("Main content and visualizations will appear here...\n\nSelect an option from the Manage Experiments menu to begin.")
        center_layout.addWidget(center_label)
        center_layout.addWidget(self.center_content)
        
        # Add top and center panes to vertical splitter
        vertical_splitter.addWidget(self.top_pane)
        vertical_splitter.addWidget(self.center_pane)
        vertical_splitter.setStretchFactor(0, 1)
        vertical_splitter.setStretchFactor(1, 3)
        
        # Add to right layout
        right_layout.addWidget(vertical_splitter)
        
        # Add left pane and right widget to main splitter
        main_splitter.addWidget(self.left_pane)
        main_splitter.addWidget(right_widget)
        
        # Set initial splitter sizes (left pane smaller)
        main_splitter.setSizes([300, 1100])
        
        # Add splitter to main layout
        main_layout.addWidget(main_splitter)
        
    def create_menu_bar(self):
        """Create the menu bar with File, Manage Experiments, and About menus."""
        menubar = self.menuBar()
        
        # FILE MENU
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Project", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Project", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # MANAGE EXPERIMENTS MENU
        experiments_menu = menubar.addMenu("Manage Experiments")
        
        # Experiment Setup
        setup_action = QAction("Experiment Setup", self)
        setup_action.triggered.connect(self.experiment_setup)
        experiments_menu.addAction(setup_action)
        
        # Analyze Data Insights
        analyze_action = QAction("Analyze Data Insights", self)
        analyze_action.triggered.connect(self.analyze_data_insights)
        experiments_menu.addAction(analyze_action)
        
        # Blueprint Repository
        blueprint_action = QAction("Blueprint Repository", self)
        blueprint_action.triggered.connect(self.blueprint_repository)
        experiments_menu.addAction(blueprint_action)
        
        # Model Leaderboard
        leaderboard_action = QAction("Model Leaderboard", self)
        leaderboard_action.triggered.connect(self.model_leaderboard)
        experiments_menu.addAction(leaderboard_action)
        
        # Experiment Insights
        insights_action = QAction("Experiment Insights", self)
        insights_action.triggered.connect(self.experiment_insights)
        experiments_menu.addAction(insights_action)
        
        # Compare Models
        compare_action = QAction("Compare Models", self)
        compare_action.triggered.connect(self.compare_models)
        experiments_menu.addAction(compare_action)
        
        # Add/Retrain Models
        retrain_action = QAction("Add/Retrain Models", self)
        retrain_action.triggered.connect(self.add_retrain_models)
        experiments_menu.addAction(retrain_action)
        
        # Edit Blueprints
        edit_blueprint_action = QAction("Edit (Composable) Blueprints", self)
        edit_blueprint_action.triggered.connect(self.edit_blueprints)
        experiments_menu.addAction(edit_blueprint_action)
        
        # ABOUT MENU
        about_menu = menubar.addMenu("About")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        about_menu.addAction(about_action)
        
        help_action = QAction("Help", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self.show_help)
        about_menu.addAction(help_action)
        
    # Menu action handlers
    def new_project(self):
        """Create a new project."""
        self.center_content.setText("New Project functionality will be implemented here.")
        
    def open_project(self):
        """Open an existing project."""
        self.center_content.setText("Open Project functionality will be implemented here.")
        
    def save_project(self):
        """Save the current project."""
        self.center_content.setText("Save Project functionality will be implemented here.")
        
    def experiment_setup(self):
        """Open experiment setup interface."""
        self.center_content.setText("Experiment Setup functionality will be implemented here.")
        
    def analyze_data_insights(self):
        """Open the Analyze Data Insights interface."""
        # Clear the center pane and replace with the data insights widget
        # Remove existing content
        center_layout = self.center_pane.layout()
        while center_layout.count():
            child = center_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Create and add the data insights widget
        insights_widget = AnalyzeDataInsights()
        center_layout.addWidget(insights_widget)
        
    def blueprint_repository(self):
        """Open blueprint repository interface."""
        self.center_content.setText("Blueprint Repository functionality will be implemented here.")
        
    def model_leaderboard(self):
        """Open model leaderboard interface."""
        self.center_content.setText("Model Leaderboard functionality will be implemented here.")
        
    def experiment_insights(self):
        """Open experiment insights interface."""
        self.center_content.setText("Experiment Insights functionality will be implemented here.")
        
    def compare_models(self):
        """Open model comparison interface."""
        self.center_content.setText("Compare Models functionality will be implemented here.")
        
    def add_retrain_models(self):
        """Open add/retrain models interface."""
        self.center_content.setText("Add/Retrain Models functionality will be implemented here.")
        
    def edit_blueprints(self):
        """Open blueprint editing interface."""
        self.center_content.setText("Edit (Composable) Blueprints functionality will be implemented here.")
        
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About MLR Working Party",
            "MLR Working Party - Experiment Manager\n\n"
            "A comprehensive GUI for managing machine learning experiments,\n"
            "analyzing data insights, and comparing model performance.\n\n"
            "Version 1.0"
        )
        
    def show_help(self):
        """Show help dialog."""
        QMessageBox.information(
            self,
            "Help",
            "MLR Working Party Help\n\n"
            "Use the 'Manage Experiments' menu to:\n"
            "- Set up new experiments\n"
            "- Analyze data insights\n"
            "- View model leaderboards\n"
            "- Compare model performance\n"
            "- Manage blueprints and feature lists\n\n"
            "For detailed documentation, please refer to the user manual."
        )


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = MLRWorkingPartyGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
