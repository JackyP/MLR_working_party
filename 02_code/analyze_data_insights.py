#!/usr/bin/env python3
"""
Analyze Data Insights Module
Provides comprehensive data analysis and visualization capabilities.
"""

import os
import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QPushButton, QLabel, QComboBox, QLineEdit,
    QGroupBox, QScrollArea, QTextEdit, QFileDialog, QMessageBox,
    QHeaderView, QListWidget, QListWidgetItem, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns


class AnalyzeDataInsights(QWidget):
    """Main widget for the Analyze Data Insights functionality."""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.feature_lists = {}  # Store custom feature lists
        self.current_dataset_path = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface with tabs for different insights."""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Analyze Data Insights")
        header_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header_label.setStyleSheet("padding: 10px; background-color: #f0f0f0;")
        layout.addWidget(header_label)
        
        # Data loading section
        load_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter dataset path or click Browse...")
        load_btn = QPushButton("Browse...")
        load_btn.clicked.connect(self.browse_dataset)
        load_data_btn = QPushButton("Load Dataset")
        load_data_btn.clicked.connect(self.load_dataset)
        
        load_layout.addWidget(QLabel("Dataset:"))
        load_layout.addWidget(self.path_input)
        load_layout.addWidget(load_btn)
        load_layout.addWidget(load_data_btn)
        layout.addLayout(load_layout)
        
        # Status label
        self.status_label = QLabel("No dataset loaded. Please load a dataset to begin analysis.")
        self.status_label.setStyleSheet("padding: 5px; color: #666;")
        layout.addWidget(self.status_label)
        
        # Create tab widget for different tiles
        self.tabs = QTabWidget()
        
        # Add tabs for each tile
        self.tabs.addTab(self.create_data_preview_tab(), "Data Preview")
        self.tabs.addTab(self.create_features_tab(), "Features")
        self.tabs.addTab(self.create_feature_lists_tab(), "Feature Lists")
        self.tabs.addTab(self.create_data_insights_tab(), "Data Insights")
        
        layout.addWidget(self.tabs)
        
    def create_data_preview_tab(self):
        """Create the Data Preview tile with visual representations."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel(
            "Data Preview: Displays a visual representation of features in your dataset, "
            "including frequent values and distributions."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("padding: 10px; background-color: #e8f4f8; border-radius: 5px;")
        layout.addWidget(desc)
        
        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Sample Size:"))
        self.preview_sample_size = QSpinBox()
        self.preview_sample_size.setRange(10, 10000)
        self.preview_sample_size.setValue(100)
        controls_layout.addWidget(self.preview_sample_size)
        
        refresh_btn = QPushButton("Refresh Preview")
        refresh_btn.clicked.connect(self.update_data_preview)
        controls_layout.addWidget(refresh_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.data_preview_layout = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        return widget
        
    def create_features_tab(self):
        """Create the Features tile with table and detailed statistics."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel(
            "Features: Displays features in a table format alongside feature importance "
            "and summary statistics. Select specific features to view more detailed data insights."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("padding: 10px; background-color: #e8f4f8; border-radius: 5px;")
        layout.addWidget(desc)
        
        # Features table
        self.features_table = QTableWidget()
        self.features_table.setColumnCount(8)
        self.features_table.setHorizontalHeaderLabels([
            "Feature", "Type", "Missing %", "Unique", "Mean", "Std", "Min", "Max"
        ])
        self.features_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.features_table.itemSelectionChanged.connect(self.on_feature_selected)
        layout.addWidget(self.features_table)
        
        # Detailed view for selected feature
        detail_group = QGroupBox("Feature Details")
        detail_layout = QVBoxLayout()
        self.feature_detail_text = QTextEdit()
        self.feature_detail_text.setReadOnly(True)
        self.feature_detail_text.setMaximumHeight(150)
        detail_layout.addWidget(self.feature_detail_text)
        detail_group.setLayout(detail_layout)
        layout.addWidget(detail_group)
        
        return widget
        
    def create_feature_lists_tab(self):
        """Create the Feature Lists tile for managing feature lists."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel(
            "Feature Lists: Create new feature lists, manage existing ones, and retrain "
            "models on different feature combinations."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("padding: 10px; background-color: #e8f4f8; border-radius: 5px;")
        layout.addWidget(desc)
        
        # Split layout for lists and management
        content_layout = QHBoxLayout()
        
        # Left side - existing feature lists
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("Existing Feature Lists:"))
        self.feature_lists_widget = QListWidget()
        self.feature_lists_widget.itemClicked.connect(self.on_feature_list_selected)
        left_layout.addWidget(self.feature_lists_widget)
        
        list_btn_layout = QHBoxLayout()
        delete_list_btn = QPushButton("Delete Selected")
        delete_list_btn.clicked.connect(self.delete_feature_list)
        list_btn_layout.addWidget(delete_list_btn)
        left_layout.addLayout(list_btn_layout)
        
        content_layout.addWidget(left_widget)
        
        # Right side - create new feature list
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("Create New Feature List:"))
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("List Name:"))
        self.new_list_name = QLineEdit()
        self.new_list_name.setPlaceholderText("Enter feature list name...")
        name_layout.addWidget(self.new_list_name)
        right_layout.addLayout(name_layout)
        
        right_layout.addWidget(QLabel("Select Features:"))
        self.available_features_list = QListWidget()
        self.available_features_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        right_layout.addWidget(self.available_features_list)
        
        create_btn = QPushButton("Create Feature List")
        create_btn.clicked.connect(self.create_new_feature_list)
        right_layout.addWidget(create_btn)
        
        content_layout.addWidget(right_widget)
        
        layout.addLayout(content_layout)
        
        # Retrain section
        retrain_group = QGroupBox("Model Retraining")
        retrain_layout = QVBoxLayout()
        retrain_desc = QLabel(
            "Select a feature list and click 'Retrain Models' to retrain all models "
            "in the experiment with the selected features."
        )
        retrain_desc.setWordWrap(True)
        retrain_layout.addWidget(retrain_desc)
        
        retrain_btn_layout = QHBoxLayout()
        self.retrain_combo = QComboBox()
        retrain_btn_layout.addWidget(QLabel("Feature List:"))
        retrain_btn_layout.addWidget(self.retrain_combo)
        retrain_btn = QPushButton("Retrain Models")
        retrain_btn.clicked.connect(self.retrain_models)
        retrain_btn_layout.addWidget(retrain_btn)
        retrain_layout.addLayout(retrain_btn_layout)
        
        retrain_group.setLayout(retrain_layout)
        layout.addWidget(retrain_group)
        
        return widget
        
    def create_data_insights_tab(self):
        """Create the Data Insights tile with feature associations."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel(
            "Data Insights: Track and visualize associations within your data using "
            "Feature Associations insight, including correlation analysis and dependencies."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("padding: 10px; background-color: #e8f4f8; border-radius: 5px;")
        layout.addWidget(desc)
        
        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Analysis Type:"))
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "Correlation Matrix",
            "Feature Associations",
            "Pairwise Correlations",
            "Feature Dependencies"
        ])
        controls_layout.addWidget(self.analysis_type_combo)
        
        analyze_btn = QPushButton("Generate Insights")
        analyze_btn.clicked.connect(self.generate_data_insights)
        controls_layout.addWidget(analyze_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Visualization area
        self.insights_canvas_widget = QWidget()
        self.insights_canvas_layout = QVBoxLayout(self.insights_canvas_widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.insights_canvas_widget)
        layout.addWidget(scroll)
        
        return widget
        
    # Data loading methods
    def browse_dataset(self):
        """Open file dialog to browse for dataset."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            os.path.expanduser("~"),
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.path_input.setText(file_path)
            
    def load_dataset(self):
        """Load the dataset from the specified path."""
        path = self.path_input.text()
        if not path:
            QMessageBox.warning(self, "Warning", "Please specify a dataset path.")
            return
            
        if not os.path.exists(path):
            QMessageBox.warning(self, "Warning", f"File not found: {path}")
            return
            
        try:
            self.data = pd.read_csv(path)
            self.current_dataset_path = path
            self.status_label.setText(
                f"Dataset loaded: {os.path.basename(path)} "
                f"({self.data.shape[0]} rows, {self.data.shape[1]} columns)"
            )
            self.status_label.setStyleSheet("padding: 5px; color: green;")
            
            # Update all tabs
            self.update_data_preview()
            self.update_features_table()
            self.update_available_features()
            
            QMessageBox.information(
                self,
                "Success",
                f"Dataset loaded successfully!\n\n"
                f"Rows: {self.data.shape[0]}\n"
                f"Columns: {self.data.shape[1]}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load dataset:\n{str(e)}"
            )
            
    # Data Preview methods
    def update_data_preview(self):
        """Update the data preview with visual representations."""
        # Clear existing content
        while self.data_preview_layout.count():
            child = self.data_preview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        if self.data is None:
            label = QLabel("No dataset loaded. Please load a dataset first.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.data_preview_layout.addWidget(label)
            return
            
        # Sample size
        sample_size = min(self.preview_sample_size.value(), len(self.data))
        sample_data = self.data.sample(n=sample_size, random_state=42) if len(self.data) > sample_size else self.data
        
        # Basic info
        info_text = f"<b>Dataset Overview</b><br>"
        info_text += f"Total Rows: {len(self.data)}<br>"
        info_text += f"Total Columns: {len(self.data.columns)}<br>"
        info_text += f"Preview Sample Size: {len(sample_data)}<br>"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("padding: 10px; background-color: #f9f9f9; border: 1px solid #ddd;")
        self.data_preview_layout.addWidget(info_label)
        
        # Create visualizations for each feature
        for col in self.data.columns[:10]:  # Limit to first 10 features
            feature_widget = self.create_feature_preview(col, sample_data)
            self.data_preview_layout.addWidget(feature_widget)
            
        self.data_preview_layout.addStretch()
        
    def create_feature_preview(self, column, sample_data):
        """Create a preview widget for a single feature."""
        group = QGroupBox(f"Feature: {column}")
        layout = QVBoxLayout()
        
        # Determine if numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(sample_data[column])
        
        info_layout = QHBoxLayout()
        
        # Statistics
        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_text.setMaximumHeight(100)
        
        if is_numeric:
            stats = sample_data[column].describe()
            stats_str = f"Type: Numeric\n"
            stats_str += f"Count: {stats['count']:.0f}\n"
            stats_str += f"Mean: {stats['mean']:.2f}\n"
            stats_str += f"Std: {stats['std']:.2f}\n"
            stats_str += f"Min: {stats['min']:.2f}\n"
            stats_str += f"Max: {stats['max']:.2f}"
        else:
            unique_count = sample_data[column].nunique()
            most_common = sample_data[column].value_counts().head(5)
            stats_str = f"Type: Categorical\n"
            stats_str += f"Unique Values: {unique_count}\n"
            stats_str += f"Most Frequent:\n"
            for val, count in most_common.items():
                stats_str += f"  {val}: {count}\n"
                
        stats_text.setText(stats_str)
        info_layout.addWidget(stats_text)
        
        # Visualization
        fig = Figure(figsize=(4, 2))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        try:
            if is_numeric:
                # Histogram for numeric
                sample_data[column].dropna().hist(ax=ax, bins=20, edgecolor='black')
                ax.set_title(f"Distribution")
            else:
                # Bar chart for categorical (top 10)
                value_counts = sample_data[column].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax)
                ax.set_title(f"Top 10 Values")
                ax.tick_params(axis='x', rotation=45)
                
            fig.tight_layout()
        except Exception as e:
            ax.text(0.5, 0.5, f"Visualization error:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            
        info_layout.addWidget(canvas)
        
        layout.addLayout(info_layout)
        group.setLayout(layout)
        
        return group
        
    # Features table methods
    def update_features_table(self):
        """Update the features table with statistics."""
        if self.data is None:
            return
            
        self.features_table.setRowCount(len(self.data.columns))
        
        for i, col in enumerate(self.data.columns):
            # Feature name
            self.features_table.setItem(i, 0, QTableWidgetItem(col))
            
            # Type
            dtype = str(self.data[col].dtype)
            self.features_table.setItem(i, 1, QTableWidgetItem(dtype))
            
            # Missing percentage
            missing_pct = (self.data[col].isna().sum() / len(self.data)) * 100
            self.features_table.setItem(i, 2, QTableWidgetItem(f"{missing_pct:.2f}%"))
            
            # Unique values
            unique = self.data[col].nunique()
            self.features_table.setItem(i, 3, QTableWidgetItem(str(unique)))
            
            # Statistics for numeric columns
            if pd.api.types.is_numeric_dtype(self.data[col]):
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                
                self.features_table.setItem(i, 4, QTableWidgetItem(f"{mean_val:.2f}"))
                self.features_table.setItem(i, 5, QTableWidgetItem(f"{std_val:.2f}"))
                self.features_table.setItem(i, 6, QTableWidgetItem(f"{min_val:.2f}"))
                self.features_table.setItem(i, 7, QTableWidgetItem(f"{max_val:.2f}"))
            else:
                for j in range(4, 8):
                    self.features_table.setItem(i, j, QTableWidgetItem("N/A"))
                    
    def on_feature_selected(self):
        """Handle feature selection in the table."""
        if self.data is None:
            return
            
        selected_items = self.features_table.selectedItems()
        if not selected_items:
            return
            
        row = selected_items[0].row()
        col_name = self.features_table.item(row, 0).text()
        
        # Generate detailed statistics
        detail_text = f"<b>Detailed Statistics for: {col_name}</b><br><br>"
        
        col_data = self.data[col_name]
        
        # Basic info
        detail_text += f"<b>Basic Information:</b><br>"
        detail_text += f"Data Type: {col_data.dtype}<br>"
        detail_text += f"Total Values: {len(col_data)}<br>"
        detail_text += f"Missing Values: {col_data.isna().sum()} ({(col_data.isna().sum()/len(col_data)*100):.2f}%)<br>"
        detail_text += f"Unique Values: {col_data.nunique()}<br><br>"
        
        if pd.api.types.is_numeric_dtype(col_data):
            # Numeric statistics
            detail_text += f"<b>Statistical Summary:</b><br>"
            detail_text += f"Mean: {col_data.mean():.4f}<br>"
            detail_text += f"Median: {col_data.median():.4f}<br>"
            detail_text += f"Mode: {col_data.mode().values[0] if len(col_data.mode()) > 0 else 'N/A'}<br>"
            detail_text += f"Std Dev: {col_data.std():.4f}<br>"
            detail_text += f"Variance: {col_data.var():.4f}<br>"
            detail_text += f"Min: {col_data.min():.4f}<br>"
            detail_text += f"25th Percentile: {col_data.quantile(0.25):.4f}<br>"
            detail_text += f"50th Percentile: {col_data.quantile(0.50):.4f}<br>"
            detail_text += f"75th Percentile: {col_data.quantile(0.75):.4f}<br>"
            detail_text += f"Max: {col_data.max():.4f}<br>"
        else:
            # Categorical statistics
            detail_text += f"<b>Value Counts (Top 10):</b><br>"
            value_counts = col_data.value_counts().head(10)
            for val, count in value_counts.items():
                pct = (count / len(col_data)) * 100
                detail_text += f"{val}: {count} ({pct:.2f}%)<br>"
                
        self.feature_detail_text.setHtml(detail_text)
        
    # Feature Lists methods
    def update_available_features(self):
        """Update the list of available features."""
        self.available_features_list.clear()
        if self.data is not None:
            for col in self.data.columns:
                self.available_features_list.addItem(col)
                
    def create_new_feature_list(self):
        """Create a new feature list from selected features."""
        name = self.new_list_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name for the feature list.")
            return
            
        if name in self.feature_lists:
            QMessageBox.warning(self, "Warning", f"A feature list named '{name}' already exists.")
            return
            
        selected_items = self.available_features_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one feature.")
            return
            
        features = [item.text() for item in selected_items]
        self.feature_lists[name] = features
        
        # Update UI
        self.feature_lists_widget.addItem(name)
        self.retrain_combo.addItem(name)
        self.new_list_name.clear()
        
        QMessageBox.information(
            self,
            "Success",
            f"Feature list '{name}' created with {len(features)} features."
        )
        
    def on_feature_list_selected(self, item):
        """Handle selection of a feature list."""
        list_name = item.text()
        if list_name in self.feature_lists:
            features = self.feature_lists[list_name]
            QMessageBox.information(
                self,
                f"Feature List: {list_name}",
                f"Features ({len(features)}):\n" + "\n".join(f"• {f}" for f in features)
            )
            
    def delete_feature_list(self):
        """Delete the selected feature list."""
        current_item = self.feature_lists_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a feature list to delete.")
            return
            
        list_name = current_item.text()
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the feature list '{list_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            del self.feature_lists[list_name]
            self.feature_lists_widget.takeItem(self.feature_lists_widget.row(current_item))
            
            # Remove from retrain combo
            index = self.retrain_combo.findText(list_name)
            if index >= 0:
                self.retrain_combo.removeItem(index)
                
    def retrain_models(self):
        """Retrain models with selected feature list."""
        feature_list_name = self.retrain_combo.currentText()
        if not feature_list_name:
            QMessageBox.warning(self, "Warning", "Please select a feature list.")
            return
            
        QMessageBox.information(
            self,
            "Retrain Models",
            f"Model retraining with feature list '{feature_list_name}' would be initiated here.\n\n"
            f"This would trigger the retraining pipeline with the selected features:\n" +
            "\n".join(f"• {f}" for f in self.feature_lists[feature_list_name])
        )
        
    # Data Insights methods
    def generate_data_insights(self):
        """Generate data insights based on selected analysis type."""
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return
            
        # Clear existing visualizations
        while self.insights_canvas_layout.count():
            child = self.insights_canvas_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        analysis_type = self.analysis_type_combo.currentText()
        
        try:
            if analysis_type == "Correlation Matrix":
                self.create_correlation_matrix()
            elif analysis_type == "Feature Associations":
                self.create_feature_associations()
            elif analysis_type == "Pairwise Correlations":
                self.create_pairwise_correlations()
            elif analysis_type == "Feature Dependencies":
                self.create_feature_dependencies()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to generate insights:\n{str(e)}"
            )
            
    def create_correlation_matrix(self):
        """Create a correlation matrix heatmap."""
        # Get numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            label = QLabel("No numeric features found in the dataset.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.insights_canvas_layout.addWidget(label)
            return
            
        # Calculate correlation
        corr_matrix = numeric_data.corr()
        
        # Create figure
        fig = Figure(figsize=(10, 8))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        self.insights_canvas_layout.addWidget(canvas)
        
        # Add interpretation text
        interpretation = QTextEdit()
        interpretation.setReadOnly(True)
        interpretation.setMaximumHeight(150)
        interpretation.setHtml(
            "<b>Interpretation:</b><br>"
            "• Values close to 1 indicate strong positive correlation<br>"
            "• Values close to -1 indicate strong negative correlation<br>"
            "• Values close to 0 indicate weak or no correlation<br>"
            "• Strong correlations may indicate redundant features or dependencies"
        )
        self.insights_canvas_layout.addWidget(interpretation)
        
    def create_feature_associations(self):
        """Create feature association visualizations."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty or len(numeric_data.columns) < 2:
            label = QLabel("Insufficient numeric features for association analysis.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.insights_canvas_layout.addWidget(label)
            return
            
        # Create multiple subplots for top correlations
        corr_matrix = numeric_data.corr()
        
        # Get top correlations (excluding diagonal)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': abs(corr_matrix.iloc[i, j])
                })
                
        corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)
        top_pairs = corr_pairs[:6]  # Top 6 correlations
        
        # Create subplots
        fig = Figure(figsize=(12, 8))
        canvas = FigureCanvas(fig)
        
        for idx, pair in enumerate(top_pairs, 1):
            ax = fig.add_subplot(2, 3, idx)
            ax.scatter(self.data[pair['feature1']], self.data[pair['feature2']], alpha=0.5)
            ax.set_xlabel(pair['feature1'])
            ax.set_ylabel(pair['feature2'])
            ax.set_title(f"Corr: {pair['correlation']:.3f}")
            
        fig.tight_layout()
        self.insights_canvas_layout.addWidget(canvas)
        
        # Add summary
        summary = QTextEdit()
        summary.setReadOnly(True)
        summary.setMaximumHeight(200)
        summary_html = "<b>Top Feature Associations:</b><br>"
        for pair in top_pairs:
            summary_html += f"• {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}<br>"
        summary.setHtml(summary_html)
        self.insights_canvas_layout.addWidget(summary)
        
    def create_pairwise_correlations(self):
        """Create pairwise correlation analysis."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty or len(numeric_data.columns) < 2:
            label = QLabel("Insufficient numeric features for pairwise analysis.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.insights_canvas_layout.addWidget(label)
            return
            
        # Calculate correlations
        results = []
        columns = numeric_data.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                data1 = numeric_data[col1].dropna()
                data2 = numeric_data[col2].dropna()
                
                # Align data
                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) > 2:
                    pearson_corr, pearson_p = pearsonr(data1[common_idx], data2[common_idx])
                    spearman_corr, spearman_p = spearmanr(data1[common_idx], data2[common_idx])
                    
                    results.append({
                        'Feature 1': col1,
                        'Feature 2': col2,
                        'Pearson': f"{pearson_corr:.3f}",
                        'Pearson p-value': f"{pearson_p:.4f}",
                        'Spearman': f"{spearman_corr:.3f}",
                        'Spearman p-value': f"{spearman_p:.4f}"
                    })
                    
        # Create table
        if results:
            results_df = pd.DataFrame(results)
            
            table = QTableWidget()
            table.setRowCount(len(results_df))
            table.setColumnCount(len(results_df.columns))
            table.setHorizontalHeaderLabels(results_df.columns.tolist())
            
            for i, row in results_df.iterrows():
                for j, val in enumerate(row):
                    table.setItem(i, j, QTableWidgetItem(str(val)))
                    
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.insights_canvas_layout.addWidget(table)
            
            # Add interpretation
            interpretation = QTextEdit()
            interpretation.setReadOnly(True)
            interpretation.setMaximumHeight(100)
            interpretation.setHtml(
                "<b>Interpretation:</b><br>"
                "• Pearson correlation measures linear relationships<br>"
                "• Spearman correlation measures monotonic relationships<br>"
                "• p-value < 0.05 indicates statistical significance"
            )
            self.insights_canvas_layout.addWidget(interpretation)
        else:
            label = QLabel("No valid pairwise correlations could be calculated.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.insights_canvas_layout.addWidget(label)
            
    def create_feature_dependencies(self):
        """Create feature dependency analysis."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            label = QLabel("No numeric features found for dependency analysis.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.insights_canvas_layout.addWidget(label)
            return
            
        # Calculate mutual information or variance inflation
        # For simplicity, we'll show correlation-based dependencies
        corr_matrix = numeric_data.corr().abs()
        
        # Create a graph-like visualization
        fig = Figure(figsize=(10, 8))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='YlOrRd', square=True, ax=ax,
                   cbar_kws={"shrink": 0.8})
        ax.set_title('Feature Dependencies (Absolute Correlations)', 
                    fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        self.insights_canvas_layout.addWidget(canvas)
        
        # Find highly dependent features
        threshold = 0.8
        dependent_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    dependent_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
                    
        if dependent_pairs:
            warning_text = QTextEdit()
            warning_text.setReadOnly(True)
            warning_text.setMaximumHeight(150)
            warning_html = "<b>⚠ Highly Dependent Features (correlation > 0.8):</b><br>"
            warning_html += "These features may be redundant and could affect model performance:<br>"
            for pair in dependent_pairs:
                warning_html += f"• {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}<br>"
            warning_text.setHtml(warning_html)
            self.insights_canvas_layout.addWidget(warning_text)


if __name__ == "__main__":
    """Test the widget standalone."""
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    widget = AnalyzeDataInsights()
    widget.show()
    sys.exit(app.exec())
