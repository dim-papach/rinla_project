"""
Report generation functionality.

This module provides tools for generating HTML and PDF reports of processing results,
allowing for easy sharing and documentation of FYF pipeline outputs.
"""

import os
from pathlib import Path
import datetime
import json
from typing import Dict, List, Optional, Any, Union
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt


class ReportGenerator:
    """
    Generate HTML and PDF reports for FYF processing results.
    
    This class creates comprehensive reports with embedded images, plots,
    and statistics to document the processing performed on astronomical images.
    
    Attributes:
        output_dir: Directory to save reports
        template_dir: Directory containing template files
    """
    
    def __init__(self, output_dir: Optional[Path] = None, 
               template_dir: Optional[Path] = None):
        """
        Initialize the ReportGenerator.
        
        Args:
            output_dir: Directory to save reports. If None, uses './reports'.
            template_dir: Directory containing template files. If None, uses
                         package default templates.
        """
        self.output_dir = Path(output_dir) if output_dir else Path('./reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # If template_dir is None, use package default templates
        if template_dir is None:
            package_dir = Path(__file__).parent.parent.parent
            self.template_dir = package_dir / 'templates'
        else:
            self.template_dir = Path(template_dir)
    
    def generate_html_report(self, 
                           title: str,
                           images: Dict[str, Path],
                           stats: Dict[str, Any],
                           output_path: Optional[Path] = None) -> Path:
        """
        Generate an HTML report with processing results.
        
        Args:
            title: Title of the report
            images: Dictionary of {image_name: image_path}
            stats: Dictionary of statistics and metadata
            output_path: Path to save the report. If None, uses output_dir/title.html.
            
        Returns:
            Path to the generated report
        """
        if output_path is None:
            output_path = self.output_dir / f"{title.replace(' ', '_')}.html"
        
        # Load HTML template
        template_path = self.template_dir / 'report_template.html'
        if template_path.exists():
            with open(template_path, 'r') as f:
                template = f.read()
        else:
            # Use basic template if file doesn't exist
            template = self._get_default_template()
        
        # Create image data
        image_html = ""
        for name, path in images.items():
            if path.exists():
                # Convert image to base64
                img_data = self._path_to_base64(path)
                image_html += f"""
                <div class="image-container">
                    <h3>{name}</h3>
                    <img src="data:image/png;base64,{img_data}" alt="{name}">
                </div>
                """
        
        # Format statistics
        stats_html = "<table class='stats-table'>\n"
        stats_html += "<tr><th>Metric</th><th>Value</th></tr>\n"
        
        for key, value in stats.items():
            if isinstance(value, dict):
                # Handle nested dictionaries
                stats_html += f"<tr><td colspan='2' class='section-header'>{key}</td></tr>\n"
                for subkey, subvalue in value.items():
                    formatted_value = f"{subvalue:.6f}" if isinstance(subvalue, float) else str(subvalue)
                    stats_html += f"<tr><td>{subkey}</td><td>{formatted_value}</td></tr>\n"
            else:
                formatted_value = f"{value:.6f}" if isinstance(value, float) else str(value)
                stats_html += f"<tr><td>{key}</td><td>{formatted_value}</td></tr>\n"
                
        stats_html += "</table>"
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Replace placeholders in template
        html_content = template.replace("{{TITLE}}", title)
        html_content = html_content.replace("{{TIMESTAMP}}", timestamp)
        html_content = html_content.replace("{{IMAGES}}", image_html)
        html_content = html_content.replace("{{STATS}}", stats_html)
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_summary_report(self, 
                              title: str,
                              results: Dict[str, Dict[str, Any]],
                              output_path: Optional[Path] = None) -> Path:
        """
        Generate a summary HTML report for multiple processed files.
        
        Args:
            title: Title of the report
            results: Dictionary of {filename: {metrics and data}}
            output_path: Path to save the report. If None, uses output_dir/title_summary.html.
            
        Returns:
            Path to the generated report
        """
        if output_path is None:
            output_path = self.output_dir / f"{title.replace(' ', '_')}_summary.html"
        
        # Load HTML template
        template_path = self.template_dir / 'summary_template.html'
        if template_path.exists():
            with open(template_path, 'r') as f:
                template = f.read()
        else:
            # Use basic template if file doesn't exist
            template = self._get_summary_template()
        
        # Create summary table
        summary_html = "<table class='summary-table'>\n"
        summary_html += "<tr><th>Filename</th><th>SSIM</th><th>MSE</th><th>MAE</th><th>Mean Residual (%)</th><th>Process Time (s)</th></tr>\n"
        
        for filename, data in results.items():
            ssim = data.get('ssim', 'N/A')
            mse = data.get('mse', 'N/A')
            mae = data.get('mae', 'N/A')
            
            residual_stats = data.get('residual_stats', {})
            mean_residual = residual_stats.get('mean', 'N/A')
            
            process_time = data.get('process_time', 'N/A')
            
            # Format values
            if isinstance(ssim, float): ssim = f"{ssim:.4f}"
            if isinstance(mse, float): mse = f"{mse:.4f}"
            if isinstance(mae, float): mae = f"{mae:.4f}"
            if isinstance(mean_residual, float): mean_residual = f"{mean_residual:.2f}"
            if isinstance(process_time, float): process_time = f"{process_time:.2f}"
            
            summary_html += f"<tr><td>{filename}</td><td>{ssim}</td><td>{mse}</td><td>{mae}</td><td>{mean_residual}</td><td>{process_time}</td></tr>\n"
        
        summary_html += "</table>"
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Replace placeholders in template
        html_content = template.replace("{{TITLE}}", title)
        html_content = html_content.replace("{{TIMESTAMP}}", timestamp)
        html_content = html_content.replace("{{SUMMARY}}", summary_html)
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    @staticmethod
    def _path_to_base64(image_path: Path) -> str:
        """
        Convert an image file to base64 for embedding in HTML.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64-encoded string representation of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    
    @staticmethod
    def _array_to_base64(array: np.ndarray) -> str:
        """
        Convert a numpy array to a base64-encoded PNG for embedding in HTML.
        
        Args:
            array: Numpy array to convert
            
        Returns:
            Base64-encoded string representation of the array as a PNG
        """
        buf = BytesIO()
        plt.imsave(buf, array, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_str
    
    @staticmethod
    def _get_default_template() -> str:
        """
        Get a default HTML template for reports.
        
        Returns:
            HTML template as a string
        """
        return """<!DOCTYPE html>
<html>
<head>
    <title>{{TITLE}}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #444;
        }
        .timestamp {
            color: #777;
            font-style: italic;
            margin-bottom: 20px;
        }
        .image-container {
            margin-bottom: 30px;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }
        .stats-table th, .stats-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .stats-table th {
            background-color: #f2f2f2;
        }
        .section-header {
            background-color: #e9e9e9;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>{{TITLE}}</h1>
    <div class="timestamp">Generated: {{TIMESTAMP}}</div>
    
    <h2>Statistics</h2>
    {{STATS}}
    
    <h2>Images</h2>
    {{IMAGES}}
</body>
</html>
"""
    
    @staticmethod
    def _get_summary_template() -> str:
        """
        Get a default HTML template for summary reports.
        
        Returns:
            HTML template as a string
        """
        return """<!DOCTYPE html>
<html>
<head>
    <title>{{TITLE}}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2 {
            color: #444;
        }
        .timestamp {
            color: #777;
            font-style: italic;
            margin-bottom: 20px;
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }
        .summary-table th, .summary-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .summary-table th {
            background-color: #f2f2f2;
        }
        .summary-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .summary-table tr:hover {
            background-color: #f1f1f1;
        }
    </style>
</head>
<body>
    <h1>{{TITLE}}</h1>
    <div class="timestamp">Generated: {{TIMESTAMP}}</div>
    
    <h2>Summary of Results</h2>
    {{SUMMARY}}
</body>
</html>
"""